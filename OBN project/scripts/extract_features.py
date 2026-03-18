import numpy as np
from scipy.fft import rfft


def extract_features(signal):

    features = []

    # time domain features
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.sum(signal**2))  # energy

    # FFT features
    fft_vals = np.abs(rfft(signal))

    features.append(np.mean(fft_vals))
    features.append(np.std(fft_vals))
    features.append(np.max(fft_vals))

    peak_freq = np.argmax(fft_vals)
    features.append(peak_freq)

    return features