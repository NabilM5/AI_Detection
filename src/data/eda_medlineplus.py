import json
import statistics


INPUT_PATH = "data/processed/medlineplus_human.jsonl"

records = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

lengths = [len(r["text"].split()) for r in records]

print("\nMEDLINEPLUS HUMAN DATASET")
print("=" * 40)
print(f"Total records: {len(records)}")
print(f"Average words: {statistics.mean(lengths):.2f}")
print(f"Min words: {min(lengths)}")
print(f"Max words: {max(lengths)}")

print("\nFirst 10 topics:")
for r in records[:10]:
    print("-", r["medical_topic"])
