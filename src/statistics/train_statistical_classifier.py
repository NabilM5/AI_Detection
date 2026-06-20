import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


CSV_PATH = "reports/statistics/text_statistics.csv"


FEATURES = [
    "word_count",
    "avg_sentence_length",
    "lexical_diversity",
    "avg_word_length",
    "punctuation_density",
]


def main():
    df = pd.read_csv(CSV_PATH)

    hc3 = df[df["dataset"] == "hc3_medicine"].copy()

    X = hc3[FEATURES]

    y = hc3["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nStatistical Feature Classifier")
    print("=" * 50)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    print("\nFeature Importance")
    print("=" * 50)

    for feature, coef in zip(FEATURES, model.coef_[0]):
        print(f"{feature:25s} {coef:.4f}")

    os.makedirs("models/statistical_classifier", exist_ok=True)

    joblib.dump(model, "models/statistical_classifier/model.pkl")
    joblib.dump(FEATURES, "models/statistical_classifier/features.pkl")

    print("\nSaved model to models/statistical_classifier/")
    print("Model classes:", model.classes_)


if __name__ == "__main__":
    main()
