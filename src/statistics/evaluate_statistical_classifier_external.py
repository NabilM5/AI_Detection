import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CSV_PATH = "reports/statistics/text_statistics.csv"

MODEL_PATH = "models/statistical_classifier/model.pkl"
FEATURES_PATH = "models/statistical_classifier/features.pkl"


def main():
    df = pd.read_csv(CSV_PATH)

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    external = df[
        (df["dataset"] == "medlineplus_human") | (df["dataset"] == "medlineplus_ai")
    ].copy()

    X = external[features]
    y_true = external["label"]

    y_pred = model.predict(X)

    print("\nExternal Statistical Classifier Evaluation")
    print("=" * 50)
    print(f"Records: {len(external)}")

    print("\nAccuracy")
    print("=" * 50)
    print(f"{accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix")
    print("=" * 50)
    print("Labels: ['ai', 'human']")
    print(confusion_matrix(y_true, y_pred, labels=["ai", "human"]))


if __name__ == "__main__":
    main()
