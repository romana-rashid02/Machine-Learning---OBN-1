from pathlib import Path
import pandas as pd
import numpy as np
import re

from data_loader import load_adc_fft
from extract_features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_day2"
OUT_CSV = BASE_DIR / "data" / "features" / "feature_dataset_day2.csv"


def extract_distance(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


features_list = []
labels = []
distances = []
source_files = []

for class_dir in RAW_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    label = class_dir.name.lower()

    print("\nProcessing class:", label)

    for file_path in class_dir.glob("*.txt"):

        print(" Reading:", file_path.name)

        data, signals, adc_signals, fft_signals = load_adc_fft(file_path)

        distance = extract_distance(file_path.name)

        for sig in adc_signals:

            feats = extract_features(sig)

            features_list.append(feats)
            labels.append(label)
            distances.append(distance)
            source_files.append(file_path.name)


features_array = np.array(features_list)

feature_names = [
    "mean","std","max","min","ptp","rms","energy","abs_mean",
    "argmax","argmin",
    "fft_mean","fft_std","fft_max","fft_peak_bin","fft_energy"
]

df = pd.DataFrame(features_array, columns=feature_names)

df["distance_cm"] = distances
df["source_file"] = source_files
df["label"] = labels

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUT_CSV, index=False)

print("\nDay 2 dataset created successfully")
print("Saved to:", OUT_CSV)
print("Shape:", df.shape)
print("\nSamples per class:")
print(df["label"].value_counts())