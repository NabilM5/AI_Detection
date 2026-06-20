import json
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append("src/eval")
from ai_confidence_score import ai_confidence


TEST_PATH = "data/processed/cross_source/test.jsonl"

MODEL_PATH = "models/cross_source_tfidf/model.pkl"
VECTORIZER_PATH = "models/cross_source_tfidf/vectorizer.pkl"

HUMAN_THRESHOLD = 0.25
AI_THRESHOLD = 0.55


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def label_from_score(score):
    if score < HUMAN_THRESHOLD:
        return "human"
    elif score < AI_THRESHOLD:
        return "uncertain"
    return "ai"


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
    print(f"Human threshold: < {HUMAN_THRESHOLD}")
    print(f"Uncertain: {HUMAN_THRESHOLD}–{AI_THRESHOLD}")
    print(f"AI threshold: >= {AI_THRESHOLD}")
    print(f"Records: {len(records)}")

    from collections import Counter

    counts = Counter(y_pred)

    print("\nPrediction Distribution")
    print("=" * 50)
    print(f"Human:    {counts.get('human', 0)}")
    print(f"Uncertain:{counts.get('uncertain', 0)}")
    print(f"AI:       {counts.get('ai', 0)}")

    auto_decided_indices = [i for i, p in enumerate(y_pred) if p != "uncertain"]

    auto_y_true = [y_true[i] for i in auto_decided_indices]
    auto_y_pred = [y_pred[i] for i in auto_decided_indices]

    print("\nAuto-Decided Subset")
    print("=" * 50)
    print(f"Auto-decided records: {len(auto_y_true)}")
    print(f"Sent to review:       {counts.get('uncertain', 0)}")

    if auto_y_true:
        print(f"Auto-decided accuracy: {accuracy_score(auto_y_true, auto_y_pred):.4f}")

        print("\nAuto-Decided Classification Report")
        print("=" * 50)
        print(classification_report(auto_y_true, auto_y_pred))

        print("\nAuto-Decided Confusion Matrix")
        print("=" * 50)
        print("Labels: ['ai', 'human']")
        print(confusion_matrix(auto_y_true, auto_y_pred, labels=["ai", "human"]))

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
