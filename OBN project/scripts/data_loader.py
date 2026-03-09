from pathlib import Path
import numpy as np


def load_adc_fft(file_path, samples_per_measurement=8192, start_index=300):
    data = np.loadtxt(file_path, dtype=str)

    cleaned = []
    for x in data.flatten():
        x = x.replace(",", ".")
        try:
            cleaned.append(float(x))
        except ValueError:
            pass

    data = np.array(cleaned, dtype=float)

    waveform = data[start_index:]
    signals = waveform.reshape(-1, samples_per_measurement)

    adc_signals = signals[0::2]
    fft_signals = signals[1::2]

    return data, signals, adc_signals, fft_signals