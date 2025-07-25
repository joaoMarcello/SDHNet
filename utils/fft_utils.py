import numpy as np
from scipy.signal import detrend

def estimate_period_fft(x, sampling_rate=1, min_period=2, max_period=None):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean()
    n = len(x)
    n2 = 1 << (n - 1).bit_length()
    fft_vals = np.fft.rfft(x, n=n2)
    power = np.abs(fft_vals)**2
    freqs = np.fft.rfftfreq(n2, d=1 / sampling_rate)
    power[0] = 0

    if max_period is None:
        max_period = n // 2
    mask = (freqs > 0) & (freqs <= 1 / min_period) & (freqs >= 1 / max_period)
    if not np.any(mask):
        return min_period

    idx = np.argmax(power[mask])
    dom_freq = freqs[mask][idx]
    period = int(round(1 / dom_freq))
    return max(period, min_period)

# def detect_period_fft(signal, top_k=1):
#     """
#     Detect the dominant period(s) of a signal using FFT.
#     """
#     signal = signal - np.mean(signal, axis=0)  # remove mean
#     fft_res = np.fft.fft(signal, axis=0)
#     freqs = np.fft.fftfreq(len(signal))

#     power = np.abs(fft_res)**2
#     half = freqs[:len(freqs)//2]
#     power = power[:len(freqs)//2]

#     avg_power = power.mean(axis=1)
#     peak_idx = np.argsort(avg_power)[-top_k:][::-1]

#     dominant_freqs = half[peak_idx]
#     dominant_periods = (1 / dominant_freqs).astype(int)
#     return dominant_periods[0] if len(dominant_periods) > 0 else None

def detect_period_fft(signal, top_k=1):
    """
    Detect dominant period(s) of a signal using FFT.
    Works with univariate or multivariate signals.
    
    Parameters:
    - signal: np.ndarray of shape (N,) or (N, D)
    - top_k: number of dominant periods to return (default=1)
    
    Returns:
    - int: dominant period if top_k=1
    - list[int]: list of top_k dominant periods if top_k > 1
    """
    signal = np.asarray(signal)

    # Garantir que é 2D: (N,) → (N, 1)
    if signal.ndim == 1:
        signal = signal[:, None]  # univariado → coluna

    # Detrending: remove tendência linear ao longo do tempo para cada variável
    signal = detrend(signal, axis=0)

    # Remove a média de cada dimensão
    signal = signal - np.mean(signal, axis=0)

    # FFT ao longo do tempo (eixo 0)
    fft_res = np.fft.fft(signal, axis=0)
    freqs = np.fft.fftfreq(signal.shape[0])

    # Usar apenas metade do espectro (frequências positivas)
    half = freqs[:len(freqs)//2]
    power = np.abs(fft_res[:len(freqs)//2, :]) ** 2

    # Média da potência entre variáveis (eixo 1)
    avg_power = power.mean(axis=1)

    # Selecionar top_k picos
    peak_idx = np.argsort(avg_power)[-top_k:][::-1]
    dominant_freqs = half[peak_idx]

    # Evitar divisão por zero
    dominant_periods = []
    for f in dominant_freqs:
        if f > 0:
            dominant_periods.append(int(round(1 / f)))

    if not dominant_periods:
        return None
    return dominant_periods[0] if top_k == 1 else dominant_periods


