from pathlib import Path
import matplotlib.pyplot as plt
from data_loader import load_adc_fft

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "laptop" / "adc_laptop1.txt"

data, signals, adc_signals, fft_signals = load_adc_fft(file_path)

print("\n----- FILE INFORMATION -----")
print("Total values in file:", len(data))
print("Detected start index:", 300)
print("Waveform samples:", signals.size)
print("Samples per measurement:", 8192)
print("Measurements detected:", len(signals))
print("Signals shape:", signals.shape)

print("ADC candidates shape:", adc_signals.shape)
print("FFT candidates shape:", fft_signals.shape)

print("\nADC candidate min/max (first 5):")
for i in range(min(5, len(adc_signals))):
    print(i, adc_signals[i].min(), adc_signals[i].max())

print("\nFFT candidate min/max (first 5):")
for i in range(min(5, len(fft_signals))):
    print(i, fft_signals[i].min(), fft_signals[i].max())

print("\nSignal 0 first 20 values:")
print(signals[0][:20])

print("\nSignal 1 first 20 values:")
print(signals[1][:20])

for i in range(min(3, len(adc_signals))):
    plt.figure()
    plt.plot(adc_signals[i])
    plt.title(f"ADC candidate {i}")

for i in range(min(3, len(fft_signals))):
    plt.figure()
    plt.plot(fft_signals[i])
    plt.title(f"FFT candidate {i}")

plt.show()