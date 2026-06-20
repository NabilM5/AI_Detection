import csv
import json
import os
import re
import string


HC3_PATH = "data/processed/hc3_medicine.jsonl"
MEDLINEPLUS_HUMAN_PATH = "data/processed/medlineplus_human.jsonl"
MEDLINEPLUS_AI_PATH = "data/processed/medlineplus_ai.jsonl"

OUTPUT_PATH = "reports/statistics/text_statistics.csv"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def get_words(text):
    return re.findall(r"\b\w+\b", text.lower())


def get_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def calculate_statistics(record, dataset_name):
    text = record["text"]

    words = get_words(text)
    sentences = get_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)

    unique_words = len(set(words))

    punctuation_count = sum(1 for ch in text if ch in string.punctuation)

    if word_count == 0:
        avg_sentence_length = 0
        lexical_diversity = 0
        avg_word_length = 0
        punctuation_density = 0
    else:
        avg_sentence_length = word_count / max(sentence_count, 1)
        lexical_diversity = unique_words / word_count
        avg_word_length = sum(len(w) for w in words) / word_count
        punctuation_density = punctuation_count / word_count

    return {
        "id": record["id"],
        "dataset": dataset_name,
        "label": record["label"],
        "medical_topic": record.get("medical_topic"),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "punctuation_density": punctuation_density,
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_rows = []

    hc3_records = load_jsonl(HC3_PATH)
    med_human_records = load_jsonl(MEDLINEPLUS_HUMAN_PATH)
    med_ai_records = load_jsonl(MEDLINEPLUS_AI_PATH)

    for record in hc3_records:
        all_rows.append(calculate_statistics(record, "hc3_medicine"))

    for record in med_human_records:
        all_rows.append(calculate_statistics(record, "medlineplus_human"))

    for record in med_ai_records:
        all_rows.append(calculate_statistics(record, "medlineplus_ai"))

    fieldnames = [
        "id",
        "dataset",
        "label",
        "medical_topic",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "lexical_diversity",
        "avg_word_length",
        "punctuation_density",
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved statistics to: {OUTPUT_PATH}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
