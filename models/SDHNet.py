import idna
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.decompose import series_decomp, series_decomp_v2
from layers.DualBlock import DoubleSample,ShortExtractor,LongExtractor, FeatureUnion


class TrendBlockLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels, hidden_dim=128):
        super(TrendBlockLinear, self).__init__()
        self.linear1 = nn.Linear(seq_len, hidden_dim)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, pred_len)
        self.channels = channels

    def forward(self, x):
        # x: (batch, seq_len, channels) → (batch * channels, seq_len)
        b, s, c = x.shape
        x = x.permute(0, 2, 1).reshape(-1, s)
        
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)

        # (batch * channels, pred_len) → (batch, pred_len, channels)
        out = out.reshape(b, c, -1).permute(0, 2, 1)
        return out
    
class TrendBlockAttention(nn.Module):
    def __init__(self, seq_len, pred_len, channels, num_heads=4):
        super(TrendBlockAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        attn_output, _ = self.attn(x, x, x)  # Self-attention
        out = self.projection(attn_output.permute(0, 2, 1))  # (b, c, seq_len) → (b, c, pred_len)
        out = out.permute(0, 2, 1)  # → (b, pred_len, c)
        return out


class NoiseBlock(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(NoiseBlock, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        self.channels = channels

    def forward(self, residual):
        # # (b, c, s) -> (b*c, s)
        # residual = residual.permute(0, 2, 1).reshape(-1, residual.shape[1])

        b, s, c = residual.shape
        residual = residual.permute(0, 2, 1).reshape(-1, s)

        out = self.project(residual)
        # (b*c, pred_len) -> (b, c, pred_len) -> (b, pred_len, c)
        out = out.reshape(-1, self.channels, out.shape[1]).permute(0, 2, 1)
        return out
    
class TrendBlock(nn.Module):
    def __init__(self, seq_len, pred_len, channels, dilation_rates=[1, 2, 4]):
        super(TrendBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.projection = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        batch, seq_len, channels = x.shape

        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)

        for conv in self.convs:
            x = torch.tanh(conv(x))

        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        x = x.reshape(batch * channels, seq_len)
        x = self.projection(x)
        x = x.reshape(batch, channels, -1)
        x = x.permute(0, 2, 1)

        return x

    # def forward(self, x):
    #     # x: (batch, seq_len, channels) → (batch, channels, seq_len)
    #     x = x.permute(0, 2, 1)
    #     for conv in self.convs:
    #         x = torch.tanh(conv(x))
    #     # (batch, channels, seq_len) → (batch, seq_len, channels)
    #     x = x.permute(0, 2, 1)
    #     return self.projection(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.dropout = nn.Dropout(p=configs.dropout)
        self.pred_len=configs.pred_len
        self.seq_len=configs.seq_len
        self.period=configs.period
        print(f">>> O periodo é {self.period}")
        self.nums=int(self.seq_len/self.period)
        if configs.features=='M':
            self.channels=configs.enc_in
        elif configs.features=='S':
            self.channels=1
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}
        self.activation=self.activations[configs.activation]

        # decomposition
        self.decompsition = series_decomp_v2(configs.moving_avg)

        # Noise
        self.noise_block = NoiseBlock(configs.seq_len, configs.pred_len, self.channels)

        # trend-cyclical prediction
        # self.trend_project=nn.Linear(in_features=configs.seq_len,out_features=configs.pred_len)
        self.trend_project = TrendBlockLinear(seq_len=configs.seq_len, pred_len=configs.pred_len, channels=self.channels)


        # self.trend_project = TrendBlock(seq_len=configs.seq_len, pred_len=configs.pred_len, channels=self.channels)

        # seasonal prediction
        self.sample=DoubleSample(self.nums)
        self.shortExtractor=ShortExtractor(configs.pooling_size,configs.seq_len,self.nums,
                                           configs.d_model,configs.dropout,configs.kernel_list,self.activation)
        self.longExtractor=LongExtractor(self.nums,configs.d_model)
        self.season_project=nn.Linear(in_features=configs.d_model,out_features=configs.pred_len) 

    def forward(self,batch_x,batch_y):

        seq_last = batch_x[:,-1:,:].detach()
        x = batch_x - seq_last
        
        # batch_x:(batch, seq_len, channel)
        season_init, trend_init, noise_init = self.decompsition(x)
        # (b,s,c)->(b,c,s)->(bc,s)
        season_init=season_init.permute(0,2,1).reshape(-1,self.seq_len)

        # # (b,s,c)->(b,c,s)->(bc,s)
        # trend_init=trend_init.permute(0,2,1).reshape(-1,self.seq_len)  

        # (bc,s)->(bc,pred_len)
        trend_out=self.trend_project(trend_init)
        #(bc,pred_len)->(b,c,pred_len)->(b,pred_len,c)                                  
        # trend_out=trend_out.reshape(-1,self.channels,self.pred_len).permute(0,2,1)

        noise_out = self.noise_block(noise_init)


        con_x,eq_x=self.sample(season_init)
        tokens=self.shortExtractor(con_x)
        h_t=self.longExtractor(eq_x)

        out=FeatureUnion(h_t,tokens)
        out=self.season_project(out)
        #(bc,pred_len)->(b,c,pred_len)->(b,pred_len,c)
        season_out=out.reshape(-1,self.channels,self.pred_len).permute(0,2,1)

        res = trend_out + season_out + noise_out
        res = res + seq_last

        return res