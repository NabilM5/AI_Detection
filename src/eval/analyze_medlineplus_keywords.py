import json
from collections import Counter

import joblib


MODEL_PATH = "models/tfidf_logreg.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MEDLINEPLUS_PATH = "data/processed/medlineplus_human.jsonl"


KEYWORDS = [
    "important",
    "healthcare",
    "provider",
    "healthcare provider",
    "treatment",
    "symptoms",
    "medical",
    "professional",
    "determine",
    "appropriate",
    "consult",
    "condition",
    "recommend",
]


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def count_keywords(text):
    text = text.lower()
    counts = {}

    for keyword in KEYWORDS:
        counts[keyword] = text.count(keyword)

    return counts


def summarize(records):
    total_docs = len(records)
    total_words = sum(len(r["text"].split()) for r in records)

    keyword_totals = Counter()

    for record in records:
        keyword_totals.update(count_keywords(record["text"]))

    print(f"Documents: {total_docs}")
    print(f"Total words: {total_words}")
    print(f"Average words: {total_words / total_docs:.2f}")

    print("\nKeyword counts per 1,000 words:")
    for keyword in KEYWORDS:
        rate = keyword_totals[keyword] / total_words * 1000
        print(f"{keyword:22s}: {keyword_totals[keyword]:4d} | {rate:.2f}")


def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    records = load_jsonl(MEDLINEPLUS_PATH)
    texts = [r["text"] for r in records]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    predicted_ai = []
    predicted_human = []

    for record, pred in zip(records, predictions):
        if pred == "ai":
            predicted_ai.append(record)
        else:
            predicted_human.append(record)

    print("\nMedlinePlus Keyword Analysis")
    print("=" * 50)

    print("\nPredicted AI group")
    print("-" * 50)
    summarize(predicted_ai)

    print("\nPredicted Human group")
    print("-" * 50)
    summarize(predicted_human)


if __name__ == "__main__":
    main()
