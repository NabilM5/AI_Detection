import json
import os

import joblib


MODEL_PATH = "models/tfidf_logreg.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MEDLINEPLUS_PATH = "data/processed/medlineplus_human.jsonl"
OUTPUT_PATH = "reports/medlineplus_false_ai_examples.txt"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def main():
    os.makedirs("reports", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    records = load_jsonl(MEDLINEPLUS_PATH)
    texts = [r["text"] for r in records]

    X = vectorizer.transform(texts)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    class_names = list(model.classes_)
    ai_index = class_names.index("ai")

    false_ai_examples = []

    for record, pred, prob in zip(records, predictions, probabilities):
        ai_probability = prob[ai_index]

        if pred == "ai":
            false_ai_examples.append(
                {
                    "topic": record["medical_topic"],
                    "url": record["url"],
                    "ai_probability": ai_probability,
                    "word_count": len(record["text"].split()),
                    "text": record["text"],
                }
            )

    false_ai_examples.sort(key=lambda x: x["ai_probability"], reverse=True)

    print("\nMedlinePlus False AI Analysis")
    print("=" * 40)
    print(f"False AI examples: {len(false_ai_examples)}")

    print("\nTop 10 most confident false AI predictions:")
    print("=" * 40)

    for example in false_ai_examples[:10]:
        print(
            f"{example['topic']} | AI prob: {example['ai_probability']:.4f} | words: {example['word_count']}"
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, example in enumerate(false_ai_examples[:30], start=1):
            f.write("=" * 80 + "\n")
            f.write(f"Example {i}\n")
            f.write(f"Topic: {example['topic']}\n")
            f.write(f"URL: {example['url']}\n")
            f.write(f"AI probability: {example['ai_probability']:.4f}\n")
            f.write(f"Word count: {example['word_count']}\n\n")
            f.write(example["text"][:1500])
            f.write("\n\n")

    print(f"\nSaved detailed examples to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
