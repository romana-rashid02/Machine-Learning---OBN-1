import numpy as np
from scipy.fft import rfft


def extract_features(signal):
    signal = np.asarray(signal, dtype=float)

    fft_vals = np.abs(rfft(signal))

    features = [
        np.mean(signal),              # mean
        np.std(signal),               # std
        np.max(signal),               # max
        np.min(signal),               # min
        np.ptp(signal),               # peak-to-peak
        np.sqrt(np.mean(signal**2)),  # rms
        np.sum(signal**2),            # energy
        np.mean(np.abs(signal)),      # abs_mean
        np.argmax(signal),            # peak index
        np.argmin(signal),            # min index

        np.mean(fft_vals),            # fft_mean
        np.std(fft_vals),             # fft_std
        np.max(fft_vals),             # fft_max
        np.argmax(fft_vals),          # fft_peak_bin
        np.sum(fft_vals**2),          # fft_energy
    ]

    return features