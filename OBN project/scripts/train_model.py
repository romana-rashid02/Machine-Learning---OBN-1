from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_CSV = BASE_DIR / "data" / "features" / "feature_dataset.csv"


def main():
    # Load feature dataset
    df = pd.read_csv(FEATURE_CSV)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print()

    # Separate features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    print("Feature matrix shape:", X.shape)
    print("Label vector shape:", y.shape)
    print("Classes:", sorted(y.unique()))
    print()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print()

    # Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    best_model_name = None
    best_model = None
    best_accuracy = -1.0

    # Train and evaluate all models
    for name, model in models.items():
        print("=" * 60)
        print(f"Training: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print("=" * 60)
    print("Best model:", best_model_name)
    print(f"Best accuracy: {best_accuracy:.4f}")

    # Optional: show feature importance for Random Forest if it wins
    if best_model_name == "Random Forest":
        print("\nFeature Importances:")
        importances = best_model.feature_importances_
        for col, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
            print(f"{col}: {imp:.4f}")


if __name__ == "__main__":
    main()