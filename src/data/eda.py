import json
import statistics


INPUT_PATH = "data/processed/hc3_medicine.jsonl"


records = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

human = [r for r in records if r["label"] == "human"]
ai = [r for r in records if r["label"] == "ai"]

human_lengths = [len(r["text"].split()) for r in human]
ai_lengths = [len(r["text"].split()) for r in ai]

print("\nDATASET OVERVIEW")
print("=" * 40)

print(f"Total records: {len(records)}")
print(f"Human: {len(human)}")
print(f"AI: {len(ai)}")

print("\nHUMAN TEXTS")
print(f"Average words: {statistics.mean(human_lengths):.2f}")
print(f"Min words: {min(human_lengths)}")
print(f"Max words: {max(human_lengths)}")

print("\nAI TEXTS")
print(f"Average words: {statistics.mean(ai_lengths):.2f}")
print(f"Min words: {min(ai_lengths)}")
print(f"Max words: {max(ai_lengths)}")
