from pathlib import Path
import re
import numpy as np
import pandas as pd
from data_loader import load_adc_fft
from extract_features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_CSV = BASE_DIR / "data" / "features" / "feature_dataset.csv"


def extract_distance(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else None


X_features = []
labels = []
distances = []
source_files = []

for class_dir in RAW_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    label = class_dir.name.lower()
    print(f"\nProcessing class: {label}")

    for file_path in class_dir.glob("*.txt"):
        print(f"  Reading file: {file_path.name}")

        try:
            data, signals, adc_signals, fft_signals = load_adc_fft(file_path)
        except Exception as e:
            print(f"  Skipping {file_path.name} because of error: {e}")
            continue

        distance = extract_distance(file_path.name)

        for sig in adc_signals:
            feats = extract_features(sig)
            X_features.append(feats)
            labels.append(label)
            distances.append(distance)
            source_files.append(file_path.name)

        print(f"  Added {len(adc_signals)} ADC measurements from {file_path.name}")

X_features = np.array(X_features)

feature_names = [
    "mean",
    "std",
    "max",
    "min",
    "ptp",
    "rms",
    "energy",
    "abs_mean",
    "argmax",
    "argmin",
    "fft_mean",
    "fft_std",
    "fft_max",
    "fft_peak_bin",
    "fft_energy",
]

df = pd.DataFrame(X_features, columns=feature_names)
df["distance_cm"] = distances
df["source_file"] = source_files
df["label"] = labels

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("\nDay-1 feature dataset built successfully")
print("Saved to:", OUT_CSV)
print("Shape:", df.shape)
print("\nSamples per class:")
print(df["label"].value_counts())
print("\nSamples per distance:")
print(df["distance_cm"].value_counts().sort_index())