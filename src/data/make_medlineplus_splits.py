import json
import os
import random
from collections import defaultdict


HUMAN_PATH = "data/processed/medlineplus_human.jsonl"
AI_PATH = "data/processed/medlineplus_ai.jsonl"
OUTPUT_DIR = "data/processed/medlineplus_splits"

RANDOM_SEED = 42
TRAIN_RATIO = 0.7


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = load_jsonl(HUMAN_PATH) + load_jsonl(AI_PATH)

    groups = defaultdict(list)
    for r in records:
        groups[r["split_group"]].append(r)

    group_ids = list(groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(group_ids)

    train_end = int(len(group_ids) * TRAIN_RATIO)

    train_groups = set(group_ids[:train_end])
    test_groups = set(group_ids[train_end:])

    train = [r for r in records if r["split_group"] in train_groups]
    test = [r for r in records if r["split_group"] in test_groups]

    write_jsonl(train, f"{OUTPUT_DIR}/train.jsonl")
    write_jsonl(test, f"{OUTPUT_DIR}/test.jsonl")

    print("MedlinePlus split finished")
    print(f"Groups: {len(group_ids)}")
    print(f"Train records: {len(train)}")
    print(f"Test records: {len(test)}")


if __name__ == "__main__":
    main()
