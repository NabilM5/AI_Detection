import json
from collections import Counter

import joblib


MODEL_PATH = "models/tfidf_logreg.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MEDLINEPLUS_AI_PATH = "data/processed/medlineplus_ai.jsonl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    records = load_jsonl(MEDLINEPLUS_AI_PATH)
    texts = [r["text"] for r in records]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    counts = Counter(predictions)
    total = len(records)

    print("\nMedlinePlus Generated AI Evaluation")
    print("=" * 50)
    print(f"Total AI records: {total}")
    print(f"Predicted AI:    {counts.get('ai', 0)}")
    print(f"Predicted Human: {counts.get('human', 0)}")

    print("\nPercentages")
    print("=" * 50)
    print(f"AI:    {counts.get('ai', 0) / total * 100:.2f}%")
    print(f"Human: {counts.get('human', 0) / total * 100:.2f}%")

    print("\nPredictions")
    print("=" * 50)
    for record, pred in zip(records, predictions):
        print(f"{record['medical_topic']} -> {pred}")


if __name__ == "__main__":
    main()
