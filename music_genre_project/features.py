import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc


def extract_features(audio_path):
    """将一首 WAV 音乐转换成 26 个特征。"""
    sample_rate, signal = wavfile.read(audio_path)
    signal = signal.astype(float)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)  # 双声道转为单声道
    signal = signal[:sample_rate * 30]
    if len(signal) < sample_rate:
        raise ValueError('音频长度至少需要 1 秒')
    peak = np.max(np.abs(signal))
    if peak == 0:
        raise ValueError('音频是静音')
    signal = signal / peak
    # FFT 长度至少覆盖一个 20 毫秒的窗口。
    nfft = max(512, 2 ** int(np.ceil(np.log2(sample_rate * 0.02))))
    coefficients = mfcc(signal, samplerate=sample_rate, numcep=13,
                        winlen=0.020, nfft=nfft, appendEnergy=False)
    values = np.concatenate([coefficients.mean(axis=0),
                             coefficients.std(axis=0)])
    if not np.isfinite(values).all():
        raise ValueError('音频产生了无效特征')
    return values
