from pathlib import Path
import numpy as np
from data_loader import load_adc_fft
from extract_features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

X = []
y = []

for class_dir in RAW_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for file_path in class_dir.glob("*.txt"):

        data, signals, adc_signals, fft_signals = load_adc_fft(file_path)

        for sig in adc_signals:
            features = extract_features(sig)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset built successfully")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", np.unique(y))