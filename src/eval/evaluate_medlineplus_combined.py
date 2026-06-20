import json

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_PATH = "models/tfidf_logreg.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

HUMAN_PATH = "data/processed/medlineplus_human.jsonl"
AI_PATH = "data/processed/medlineplus_ai.jsonl"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    human_records = load_jsonl(HUMAN_PATH)
    ai_records = load_jsonl(AI_PATH)

    records = human_records + ai_records

    X = vectorizer.transform([r["text"] for r in records])
    y_true = [r["label"] for r in records]

    y_pred = model.predict(X)

    print("\nCombined MedlinePlus External Evaluation")
    print("=" * 50)
    print(f"Human records: {len(human_records)}")
    print(f"AI records:    {len(ai_records)}")
    print(f"Total:         {len(records)}")

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
