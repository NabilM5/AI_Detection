import joblib
import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


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

X_train = [r["text"] for r in train_records]
y_train = [r["label"] for r in train_records]

X_test = [r["text"] for r in test_records]
y_test = [r["label"] for r in test_records]

vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]

top_ai = coefficients.argsort()[-20:]
top_human = coefficients.argsort()[:20]

print("\nTOP AI FEATURES")
for idx in reversed(top_ai):
    print(feature_names[idx], coefficients[idx])

print("\nTOP HUMAN FEATURES")
for idx in top_human:
    print(feature_names[idx], coefficients[idx])

print("\nAccuracy")
print("=" * 40)
print(accuracy_score(y_test, predictions))

print("\nClassification Report")
print("=" * 40)
print(classification_report(y_test, predictions))


os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/tfidf_logreg.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved.")
