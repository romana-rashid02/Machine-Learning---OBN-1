from pathlib import Path
import numpy as np


def load_adc_fft(file_path, samples_per_measurement=8192, start_index=300):
    """
    Load one txt file and split it into ADC and FFT measurements.
    This version is faster and more robust than np.loadtxt() for large files.
    """
    cleaned = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            for x in parts:
                x = x.strip().replace(",", ".")
                if not x:
                    continue
                try:
                    cleaned.append(float(x))
                except ValueError:
                    continue

    data = np.array(cleaned, dtype=float)

    if len(data) <= start_index:
        raise ValueError(f"File too short after reading: {file_path}")

    waveform = data[start_index:]

    usable_length = (len(waveform) // samples_per_measurement) * samples_per_measurement
    waveform = waveform[:usable_length]

    if usable_length == 0:
        raise ValueError(f"No usable waveform data in file: {file_path}")

    signals = waveform.reshape(-1, samples_per_measurement)

    adc_signals = signals[0::2]
    fft_signals = signals[1::2]

    return data, signals, adc_signals, fft_signals