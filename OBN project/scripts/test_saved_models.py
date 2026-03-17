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
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # Keep distance as feature, same as training
    X = df.drop(columns=["label", "source_file"])
    y = df["label"]

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nClass counts:")
    print(y.value_counts())

    # Find latest saved models
    model_files = {
        "Random_Forest": get_latest_model(MODEL_DIR, "Random_Forest"),
        "SVM": get_latest_model(MODEL_DIR, "SVM"),
        "KNN": get_latest_model(MODEL_DIR, "KNN"),
    }

    # Test each model
    for name, model_path in model_files.items():
        print("\n" + "=" * 60)
        print(f"Testing: {name}")
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


if __name__ == "__main__":
    main()