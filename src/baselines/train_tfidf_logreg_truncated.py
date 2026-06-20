import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/splits/train.jsonl"
TEST_PATH = "data/processed/splits/test.jsonl"
MAX_WORDS = 100


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def truncate_text(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])


train_records = load_jsonl(TRAIN_PATH)
test_records = load_jsonl(TEST_PATH)

X_train = [truncate_text(r["text"], MAX_WORDS) for r in train_records]
y_train = [r["label"] for r in train_records]

X_test = [truncate_text(r["text"], MAX_WORDS) for r in test_records]
y_test = [r["label"] for r in test_records]

vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("\nTF-IDF + Logistic Regression Truncated")
print("=" * 50)
print(f"Max words per text: {MAX_WORDS}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification Report")
print("=" * 50)
print(classification_report(y_test, predictions))
