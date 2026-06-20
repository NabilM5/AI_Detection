import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/splits/train.jsonl"
TEST_PATH = "data/processed/splits/test.jsonl"

MAX_WORDS_LIST = [10, 25, 50, 100, 200, None]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def truncate_text(text, max_words):
    if max_words is None:
        return text

    words = text.split()
    return " ".join(words[:max_words])


def run_experiment(train_records, test_records, max_words):
    X_train = [truncate_text(r["text"], max_words) for r in train_records]
    y_train = [r["label"] for r in train_records]

    X_test = [truncate_text(r["text"], max_words) for r in test_records]
    y_test = [r["label"] for r in test_records]

    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy, classification_report(y_test, predictions)


def main():
    train_records = load_jsonl(TRAIN_PATH)
    test_records = load_jsonl(TEST_PATH)

    print("\nTF-IDF Truncation Sweep")
    print("=" * 50)

    results = []

    for max_words in MAX_WORDS_LIST:
        label = "full" if max_words is None else str(max_words)

        accuracy, report = run_experiment(
            train_records=train_records, test_records=test_records, max_words=max_words
        )

        results.append((label, accuracy))

        print(f"\nMax words: {label}")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)

    print("\nSummary")
    print("=" * 50)
    print("Words kept | Accuracy")
    print("-" * 25)

    for label, accuracy in results:
        print(f"{label:10s} | {accuracy:.4f}")


if __name__ == "__main__":
    main()
