import json
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append("src/eval")
from ai_confidence_score import ai_confidence


TEST_PATH = "data/processed/cross_source/test.jsonl"

MODEL_PATH = "models/cross_source_tfidf/model.pkl"
VECTORIZER_PATH = "models/cross_source_tfidf/vectorizer.pkl"

CONFIDENCE_THRESHOLD = 0.30


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def label_from_score(score):
    if score >= CONFIDENCE_THRESHOLD:
        return "ai"
    return "human"


def main():
    records = load_jsonl(TEST_PATH)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    texts = [r["text"] for r in records]
    y_true = [r["label"] for r in records]

    X = vectorizer.transform(texts)

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    ai_index = classes.index("ai")

    ai_probs = probabilities[:, ai_index]

    y_pred = []
    scored_examples = []

    for record, ai_prob in zip(records, ai_probs):
        score = ai_confidence(ai_prob, record["text"])
        pred = label_from_score(score)

        y_pred.append(pred)

        scored_examples.append(
            {
                "topic": record.get("medical_topic"),
                "true_label": record["label"],
                "prediction": pred,
                "model_ai_probability": float(ai_prob),
                "confidence_score": float(score),
                "word_count": len(record["text"].split()),
            }
        )

    print("\nAI Confidence Score Evaluation")
    print("=" * 50)
    print(f"Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Records: {len(records)}")

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

    print("\nTop 10 highest confidence AI scores")
    print("=" * 50)

    scored_examples.sort(key=lambda x: x["confidence_score"], reverse=True)

    for ex in scored_examples[:10]:
        print(
            f"{ex['topic']} | true={ex['true_label']} | pred={ex['prediction']} | "
            f"prob={ex['model_ai_probability']:.3f} | score={ex['confidence_score']:.3f} | "
            f"words={ex['word_count']}"
        )


if __name__ == "__main__":
    main()
