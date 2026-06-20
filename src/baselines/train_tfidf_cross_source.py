import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


TRAIN_PATH = "data/processed/cross_source/train.jsonl"
TEST_PATH = "data/processed/cross_source/test.jsonl"

MODEL_DIR = "models/cross_source_tfidf"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def print_top_features(model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    classes = list(model.classes_)
    print("\nModel classes")
    print("=" * 50)
    print(classes)

    positive_class = classes[1]
    negative_class = classes[0]

    top_positive = np.argsort(coefficients)[-20:][::-1]
    top_negative = np.argsort(coefficients)[:20]

    print(f"\nTop features for class: {positive_class}")
    print("=" * 50)
    for idx in top_positive:
        print(f"{feature_names[idx]:30s} {coefficients[idx]:.4f}")

    print(f"\nTop features for class: {negative_class}")
    print("=" * 50)
    for idx in top_negative:
        print(f"{feature_names[idx]:30s} {coefficients[idx]:.4f}")


def main():
    train_records = load_jsonl(TRAIN_PATH)
    test_records = load_jsonl(TEST_PATH)

    X_train = [r["text"] for r in train_records]
    y_train = [r["label"] for r in train_records]

    X_test = [r["text"] for r in test_records]
    y_test = [r["label"] for r in test_records]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    probabilities = model.predict_proba(X_test_vec)

    classes = list(model.classes_)
    ai_index = classes.index("ai")

    y_test_binary = [1 if label == "ai" else 0 for label in y_test]
    ai_probabilities = probabilities[:, ai_index]

    roc_auc = roc_auc_score(y_test_binary, ai_probabilities)

    print("\nCross-Source TF-IDF Evaluation")
    print("=" * 50)
    print(f"Train records: {len(train_records)}")
    print(f"Test records:  {len(test_records)}")

    print("\nAccuracy")
    print("=" * 50)
    print(f"{accuracy_score(y_test, predictions):.4f}")

    print("\nROC-AUC")
    print("=" * 50)
    print(f"{roc_auc:.4f}")

    print("\nClassification Report")
    print("=" * 50)
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print("=" * 50)
    print("Labels: ['ai', 'human']")
    print(confusion_matrix(y_test, predictions, labels=["ai", "human"]))

    print_top_features(model, vectorizer)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

    print(f"\nSaved model to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
