from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    FEATURE_CSV = BASE_DIR / "data" / "features" / "feature_dataset.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    # Load Day 1 dataset
    df = pd.read_csv(FEATURE_CSV)

    # ✅ Filter to 4 distances only
    df = df[df['distance_cm'].isin([10, 15, 20, 25])]

    print("Dataset loaded and filtered to 4 distances")
    print("Shape after filtering:", df.shape)
    print("\nSamples per class:")
    print(df["label"].value_counts())
    print("\nSamples per distance:")
    print(df["distance_cm"].value_counts().sort_index())

    X = df.drop(columns=["label", "source_file"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    models = {
        "Random_Forest_4dist": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        "SVM_4dist": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
        ]),
        "KNN_4dist": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ])
    }

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", round(acc, 4))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # ✅ Save with _4dist_ in filename so it won't be confused with full-range models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = MODEL_DIR / f"{name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print("Model saved to:", model_path)


if __name__ == "__main__":
    main()