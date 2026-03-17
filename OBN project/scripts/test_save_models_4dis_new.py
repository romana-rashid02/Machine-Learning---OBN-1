from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_latest_model(model_dir, prefix):
    files = sorted(model_dir.glob(f"{prefix}_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No saved model found for {prefix}")
    return files[-1]


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"
    TEST_CSV = BASE_DIR / "data" / "features" / "feature_dataset_day2.csv"

    # Load and filter Day 2 data to 4 distances
    df = pd.read_csv(TEST_CSV)
    df = df[df['distance_cm'].isin([10, 15, 20, 25])]

    print("Day-2 dataset loaded and filtered to 4 distances")
    print("Shape:", df.shape)
    print("\nClass counts:")
    print(df["label"].value_counts())
    print("\nSamples per distance:")
    print(df["distance_cm"].value_counts().sort_index())

    X = df.drop(columns=["label", "source_file"])
    y = df["label"]

    # ✅ Load the 4-distance specific models
    model_files = {
        "Random_Forest_4dist": get_latest_model(MODEL_DIR, "Random_Forest_4dist"),
        "SVM_4dist": get_latest_model(MODEL_DIR, "SVM_4dist"),
        "KNN_4dist": get_latest_model(MODEL_DIR, "KNN_4dist"),
    }

    for name, model_path in model_files.items():
        print("\n" + "=" * 60)
        print(f"Testing (4-dist model): {name}")
        print("=" * 60)
        print("Loaded model:", model_path)

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        print("Accuracy:", round(acc, 4))

        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        # Per-distance accuracy
        print("\nAccuracy per distance:")
        for dist in [10, 15, 20, 25]:
            mask = df["distance_cm"] == dist
            X_dist = X[mask]
            y_dist = y[mask]
            y_pred_dist = model.predict(X_dist)
            acc_dist = accuracy_score(y_dist, y_pred_dist)
            print(f"  {dist} cm : {round(acc_dist, 4)}")


if __name__ == "__main__":
    main()