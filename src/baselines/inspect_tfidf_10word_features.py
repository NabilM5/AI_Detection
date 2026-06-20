import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = "data/processed/splits/train.jsonl"
MAX_WORDS = 10


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def truncate_text(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])


records = load_jsonl(TRAIN_PATH)

X_train = [truncate_text(r["text"], MAX_WORDS) for r in records]
y_train = [r["label"] for r in records]

vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

print("Model classes:", model.classes_)

print("\nTop positive features")
print("=" * 40)
for idx in coefficients.argsort()[-30:][::-1]:
    print(f"{feature_names[idx]:25s} {coefficients[idx]:.4f}")

print("\nTop negative features")
print("=" * 40)
for idx in coefficients.argsort()[:30]:
    print(f"{feature_names[idx]:25s} {coefficients[idx]:.4f}")
