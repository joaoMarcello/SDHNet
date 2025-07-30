import os
import torch
import numpy as np

from data_provider.data_factory import data_provider
from utils.fft_utils import detect_period_fft, fft_detect_period_paper_style, fft_detect_period_paper_style_v2

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        if self.args.period <= 0:
            window = self.args.seq_len
            dataset, _ = data_provider(self.args, flag='train')
            data_x = dataset.data_x
            data_x = dataset.data_x[:window].copy()
            # self.args.period = detect_period_fft(data_x)
            self.args.period = fft_detect_period_paper_style_v2(data_x)
            print(f"Período detectado por FFT: {self.args.period}")

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(                                
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices 
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')  
            print('Use CPU')
        return device                     


    def _get_data(self):
        pass                              
                                          
    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
