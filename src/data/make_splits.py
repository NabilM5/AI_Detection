import json
import os
import random
from collections import defaultdict


INPUT_PATH = "data/processed/hc3_medicine.jsonl"
OUTPUT_DIR = "data/processed/splits"
RANDOM_SEED = 42


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = read_jsonl(INPUT_PATH)

    groups = defaultdict(list)
    for record in records:
        groups[record["split_group"]].append(record)

    group_ids = list(groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(group_ids)

    n = len(group_ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_groups = set(group_ids[:train_end])
    val_groups = set(group_ids[train_end:val_end])
    test_groups = set(group_ids[val_end:])

    train = [r for r in records if r["split_group"] in train_groups]
    val = [r for r in records if r["split_group"] in val_groups]
    test = [r for r in records if r["split_group"] in test_groups]

    write_jsonl(train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    write_jsonl(val, os.path.join(OUTPUT_DIR, "val.jsonl"))
    write_jsonl(test, os.path.join(OUTPUT_DIR, "test.jsonl"))

    print("Split finished")
    print(f"Groups: {n}")
    print(f"Train: {len(train)} records")
    print(f"Val:   {len(val)} records")
    print(f"Test:  {len(test)} records")


if __name__ == "__main__":
    main()
