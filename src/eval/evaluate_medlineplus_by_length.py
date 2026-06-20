import json
from collections import defaultdict

import joblib


MODEL_PATH = "models/tfidf_logreg.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MEDLINEPLUS_PATH = "data/processed/medlineplus_human.jsonl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_length_bucket(word_count):
    if word_count < 100:
        return "0-99"
    if word_count < 200:
        return "100-199"
    if word_count < 400:
        return "200-399"
    if word_count < 700:
        return "400-699"
    return "700+"


def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    records = load_jsonl(MEDLINEPLUS_PATH)
    texts = [r["text"] for r in records]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    buckets = defaultdict(lambda: {"total": 0, "ai": 0, "human": 0})

    for record, pred in zip(records, predictions):
        word_count = len(record["text"].split())
        bucket = get_length_bucket(word_count)

        buckets[bucket]["total"] += 1
        buckets[bucket][pred] += 1

    print("\nMedlinePlus Predictions by Length")
    print("=" * 50)

    for bucket in ["0-99", "100-199", "200-399", "400-699", "700+"]:
        total = buckets[bucket]["total"]
        ai = buckets[bucket]["ai"]
        human = buckets[bucket]["human"]

        if total == 0:
            continue

        ai_percent = ai / total * 100
        human_percent = human / total * 100

        print(f"\nLength bucket: {bucket} words")
        print(f"Total: {total}")
        print(f"Predicted AI: {ai} ({ai_percent:.2f}%)")
        print(f"Predicted human: {human} ({human_percent:.2f}%)")


if __name__ == "__main__":
    main()
