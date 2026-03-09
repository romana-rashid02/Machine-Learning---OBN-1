from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "steel" / "adc_steel1.txt"

data = np.loadtxt(file_path, dtype=str)

cleaned = []
for x in data.flatten():
    x = x.replace(",", ".")
    try:
        cleaned.append(float(x))
    except:
        pass

data = np.array(cleaned)

print("Total values:", len(data))
print("First 25 values:", data[:25])

waveform = data[16:]
print("Waveform length:", len(waveform))

samples_per_measurement = 32768
usable = waveform[: (len(waveform) // samples_per_measurement) * samples_per_measurement]
signals = usable.reshape(-1, samples_per_measurement)

print("Number of measurements:", len(signals))

plt.figure()
plt.plot(waveform[:5000])
plt.title("Steel ADC waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.figure()
plt.plot(signals[0])
plt.title("Single steel measurement")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.show()