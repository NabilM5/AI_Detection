import json

import joblib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


TEST_PATH = "data/processed/cross_source/test.jsonl"
MODEL_PATH = "models/cross_source_tfidf/model.pkl"
VECTORIZER_PATH = "models/cross_source_tfidf/vectorizer.pkl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    records = load_jsonl(TEST_PATH)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X = vectorizer.transform([r["text"] for r in records])
    y_true = [r["label"] for r in records]

    probs = model.predict_proba(X)
    classes = list(model.classes_)
    ai_index = classes.index("ai")

    ai_probs = probs[:, ai_index]

    print("\nCross-Source Threshold Tuning")
    print("=" * 70)
    print("threshold | accuracy | ai_precision | ai_recall | ai_f1 | human_recall")
    print("-" * 70)

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        y_pred = ["ai" if p >= threshold else "human" for p in ai_probs]

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=["ai", "human"],
            zero_division=0,
        )

        ai_precision = precision[0]
        ai_recall = recall[0]
        ai_f1 = f1[0]
        human_recall = recall[1]

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:8.4f} | "
            f"{ai_precision:12.4f} | "
            f"{ai_recall:9.4f} | "
            f"{ai_f1:5.4f} | "
            f"{human_recall:12.4f}"
        )


if __name__ == "__main__":
    main()
