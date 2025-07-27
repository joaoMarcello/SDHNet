import numpy as np
from scipy.signal import detrend
from scipy.fft import fft

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

def detect_dominant_period(data, sampling_rate=1, fallback_value=24):
    """
    Detecta o período dominante de um sinal temporal (univariado ou multivariado) usando FFT.
    
    Args:
        data (np.ndarray): shape (T,) ou (B, T, C) — série temporal
        sampling_rate (float): taxa de amostragem
        fallback_value (int): valor retornado se período não puder ser detectado

    Returns:
        int: período dominante estimado
    """
    # Garante 3D: (B, T, C)
    if data.ndim == 1:
        data = data[None, :, None]
    elif data.ndim == 2:
        data = data[:, :, None]

    B, T, C = data.shape
    periods = []

    for b in range(B):
        for c in range(C):
            signal = data[b, :, c]
            signal = signal - np.mean(signal)  # remove tendência

            if np.allclose(signal, 0, atol=1e-6):
                continue  # série constante

            # FFT
            fft_vals = fft(signal)
            mags = np.abs(fft_vals)[:T // 2]
            mags[0] = 0  # remove DC

            if np.all(mags < 1e-6):
                continue  # nada significativo

            freqs = np.fft.fftfreq(T, d=sampling_rate)[:T // 2]
            dominant_idx = np.argmax(mags)
            dominant_freq = freqs[dominant_idx]

            if dominant_freq > 0:
                period = 1 / dominant_freq
                periods.append(period)

    if not periods:
        return fallback_value

    # Round to nearest int and return most common or average
    return int(round(np.median(periods)))

def fft_detect_period(x, seq_len=720, fallback=24):
    """
    Detecta o período dominante de uma série temporal usando FFT,
    como descrito no paper do SDHNet, com proteções para casos reais.

    Parâmetros:
    - x: ndarray (1D ou 2D). Se 2D, considera multivariada (time, channels).
    - seq_len: int, tamanho da janela FFT (opcional, default: len(x))
    - fallback: int, valor de período a retornar se detecção falhar (opcional, default: seq_len)

    Retorno:
    - int: período dominante estimado
    """

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]  # transforma em multivariada com 1 canal

    T = seq_len or x.shape[0]
    fallback = fallback or T

    periods = []
    for i in range(x.shape[1]):
        series = x[:, i]
        if np.allclose(series, series[0]):
            continue  # série constante, sem informação espectral

        # Remoção de tendência (como no paper: x - x̄)
        series = series - np.mean(series)

        # FFT e espectro de magnitude
        fft_result = np.fft.fft(series, n=T)
        mag = np.abs(fft_result)[:T // 2]
        mag[0] = 0  # ignora componente DC

        if np.allclose(mag, 0):
            continue  # espectro plano

        k = np.argmax(mag)
        if k == 0:
            continue  # nenhuma frequência útil

        period = T // k
        periods.append(period)

    if not periods:
        return fallback

    return int(np.median(periods))

def fft_detect_period_paper_style(x, seq_len=None, top_k=3, fallback=24):
    """
    Implementa a detecção de período como descrito no paper SDHNet (Eq. 3),
    usando FFT com top-k amplitudes e seleção do maior período correspondente.
    """

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]

    T = seq_len or x.shape[0]
    fallback = fallback or T

    periods = []

    for i in range(x.shape[1]):
        series = x[:, i]
        if np.allclose(series, series[0]):
            continue

        series = series - np.mean(series)

        fft_result = np.fft.fft(series, n=T)
        mag = np.abs(fft_result)[:T // 2]
        mag[0] = 0  # remove DC

        top_indices = np.argpartition(mag, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-mag[top_indices])]

        top_periods = [int(np.ceil(T / k)) for k in top_indices if k > 0]
        if top_periods:
            periods.append(max(top_periods))

    return int(np.median(periods)) if periods else fallback

from scipy.fft import fft, fftfreq

def fft_detect_period_paper_style_v2(signal, top_k=3):
    """
    Detect dominant period(s) of a signal using FFT (SciPy version).
    Works with univariate or multivariate signals.
    Returns the largest period among the top-k amplitude peaks.

    Parameters:
    - signal: np.ndarray of shape (N,) or (N, D)
    - top_k: number of dominant periods to consider (default=3)

    Returns:
    - int: dominant period (largest among top_k)
    """
    signal = np.asarray(signal)
    
    # Garantir que o sinal seja 2D (N, D)
    if signal.ndim == 1:
        signal = signal[:, None]
    
    # Remover média para cada dimensão
    signal = signal - np.mean(signal, axis=0)
    
    N = signal.shape[0]
    fft_res = fft(signal, axis=0)
    freqs = fftfreq(N)
    
    # Usar somente as frequências positivas
    half = freqs[:N // 2]
    power = np.abs(fft_res[:N // 2, :]) ** 2
    
    # Média da potência nas dimensões
    avg_power = power.mean(axis=1)
    
    # Pegar os índices dos top_k maiores amplitudes
    peak_idx = np.argsort(avg_power)[-top_k:][::-1]
    dominant_freqs = half[peak_idx]
    
    # Calcular períodos correspondentes (evitar freq=0)
    dominant_periods = []
    for f in dominant_freqs:
        if f > 0:
            dominant_periods.append(int(round(1 / f)))
    
    if not dominant_periods:
        return None
    
    # Retornar o maior período, conforme o paper
    return max(dominant_periods)
