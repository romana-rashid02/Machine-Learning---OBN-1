from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_adc_fft
from extract_features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

X_features = []
y = []

for class_dir in RAW_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for file_path in class_dir.glob("*.txt"):
        data, signals, adc_signals, fft_signals = load_adc_fft(file_path)

        for sig in adc_signals:
            feats = extract_features(sig)
            X_features.append(feats)
            y.append(label)

X_features = np.array(X_features)
y = np.array(y)

print("Feature dataset built successfully")
print("X_features shape:", X_features.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))

feature_names = [
    "mean",
    "std",
    "max",
    "min",
    "energy",
    "fft_mean",
    "fft_std",
    "fft_max",
    "fft_peak_bin",
]

df = pd.DataFrame(X_features, columns=feature_names)
df["label"] = y

out_path = BASE_DIR / "data" / "features" / "feature_dataset.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print("Saved feature CSV to:", out_path)