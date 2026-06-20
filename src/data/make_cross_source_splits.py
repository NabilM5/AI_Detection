import json
import os


HC3_TRAIN_PATH = "data/processed/splits/train.jsonl"
MEDLINEPLUS_TRAIN_PATH = "data/processed/medlineplus_splits/train.jsonl"
MEDLINEPLUS_TEST_PATH = "data/processed/medlineplus_splits/test.jsonl"
OUTPUT_DIR = "data/processed/cross_source"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    hc3_train = load_jsonl(HC3_TRAIN_PATH)
    medlineplus_train = load_jsonl(MEDLINEPLUS_TRAIN_PATH)
    medlineplus_test = load_jsonl(MEDLINEPLUS_TEST_PATH)

    train_records = hc3_train + medlineplus_train
    test_records = medlineplus_test

    train_groups = {record["split_group"] for record in train_records}
    test_groups = {record["split_group"] for record in test_records}
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f"Detected {len(overlap)} split groups in both train and test sets."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_jsonl(train_records, os.path.join(OUTPUT_DIR, "train.jsonl"))
    write_jsonl(test_records, os.path.join(OUTPUT_DIR, "test.jsonl"))

    print("Cross-source split finished")
    print(f"Train records: {len(train_records)}")
    print(f"Test records:  {len(test_records)}")
    print(f"Group overlap: {len(overlap)}")


if __name__ == "__main__":
    main()
