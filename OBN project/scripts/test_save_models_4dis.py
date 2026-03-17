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

    # Load day-2 dataset
    df = pd.read_csv(TEST_CSV)

    print("Day-2 dataset loaded successfully")
    print("Path:", TEST_CSV)
    print("Shape before filtering:", df.shape)

    # ✅ CHANGE 1 — Filter to 4 distances only
    df = df[df['distance_cm'].isin([10, 15, 20, 25])]
    print("Shape after filtering to 10, 15, 20, 25 cm:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # Keep distance as feature, same as training
    X = df.drop(columns=["label", "source_file"])
    y = df["label"]

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nClass counts:")
    print(y.value_counts())

    # ✅ CHANGE 2 — Print sample counts per distance to verify filtering worked
    print("\nSamples per distance:")
    print(df["distance_cm"].value_counts().sort_index())

    # Find latest saved models
    model_files = {
        "Random_Forest": get_latest_model(MODEL_DIR, "Random_Forest"),
        "SVM": get_latest_model(MODEL_DIR, "SVM"),
        "KNN": get_latest_model(MODEL_DIR, "KNN"),
    }

    # Test each model
    for name, model_path in model_files.items():
        print("\n" + "=" * 60)
        # ✅ CHANGE 3 — Updated title to make clear this is 4-distance test
        print(f"Testing (4 distances): {name}")
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

        # ✅ CHANGE 4 — Per-distance accuracy breakdown
        print("\nAccuracy per distance:")
        for dist in [10, 15, 20, 25]:
            mask = df["distance_cm"] == dist
            if mask.sum() == 0:
                continue
            X_dist = X[mask]
            y_dist = y[mask]
            y_pred_dist = model.predict(X_dist)
            acc_dist = accuracy_score(y_dist, y_pred_dist)
            print(f"  {dist} cm : {round(acc_dist, 4)}")


if __name__ == "__main__":
    main()