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

    probabilities = model.predict_proba(X)
    class_names = list(model.classes_)
    ai_index = class_names.index("ai")

    buckets = defaultdict(list)

    for record, prob in zip(records, probabilities):
        word_count = len(record["text"].split())
        bucket = get_length_bucket(word_count)
        ai_probability = prob[ai_index]

        buckets[bucket].append(ai_probability)

    print("\nMedlinePlus Length vs AI Probability")
    print("=" * 50)

    for bucket in ["0-99", "100-199", "200-399", "400-699", "700+"]:
        probs = buckets[bucket]

        if not probs:
            continue

        avg_prob = sum(probs) / len(probs)
        min_prob = min(probs)
        max_prob = max(probs)

        print(f"\nLength bucket: {bucket} words")
        print(f"Documents: {len(probs)}")
        print(f"Average AI probability: {avg_prob:.4f}")
        print(f"Min AI probability: {min_prob:.4f}")
        print(f"Max AI probability: {max_prob:.4f}")


if __name__ == "__main__":
    main()
