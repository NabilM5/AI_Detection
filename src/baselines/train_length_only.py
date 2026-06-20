import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


TRAIN_PATH = "data/processed/splits/train.jsonl"
TEST_PATH = "data/processed/splits/test.jsonl"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


train_records = load_jsonl(TRAIN_PATH)
test_records = load_jsonl(TEST_PATH)

X_train = [[len(r["text"].split())] for r in train_records]
y_train = [r["label"] for r in train_records]

X_test = [[len(r["text"].split())] for r in test_records]
y_test = [r["label"] for r in test_records]

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nLength-Only Baseline")
print("=" * 40)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification Report")
print("=" * 40)
print(classification_report(y_test, predictions))

print("\nModel coefficient")
print("=" * 40)
print(f"Coefficient: {model.coef_[0][0]:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")
