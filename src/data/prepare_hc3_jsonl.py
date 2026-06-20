import json
import os


RAW_PATH = "data/raw/hc3_medicine_raw.json"
OUTPUT_PATH = "data/processed/hc3_medicine.jsonl"


def clean_text(text):
    if text is None:
        return ""
    return " ".join(text.strip().split())


def main():
    os.makedirs("data/processed", exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []

    for i, item in enumerate(raw_data):
        question = clean_text(item.get("question", ""))
        human_answers = item.get("human_medical_answers", [])
        ai_answers = item.get("chatgpt_medical_answers", [])

        source_document_id = f"hc3_medicine_{i}"
        split_group = source_document_id

        for j, answer in enumerate(human_answers):
            text = clean_text(answer)
            if not text:
                continue

            records.append(
                {
                    "id": f"{source_document_id}_human_{j}",
                    "text": text,
                    "label": "human",
                    "question": question,
                    "source_dataset": "hc3",
                    "source_document_id": source_document_id,
                    "genre": "medical_qa",
                    "generator": None,
                    "split_group": split_group,
                }
            )

        for j, answer in enumerate(ai_answers):
            text = clean_text(answer)
            if not text:
                continue

            records.append(
                {
                    "id": f"{source_document_id}_ai_{j}",
                    "text": text,
                    "label": "ai",
                    "question": question,
                    "source_dataset": "hc3",
                    "source_document_id": source_document_id,
                    "genre": "medical_qa",
                    "generator": "chatgpt",
                    "split_group": split_group,
                }
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {OUTPUT_PATH}")
    print(f"Human: {sum(r['label'] == 'human' for r in records)}")
    print(f"AI: {sum(r['label'] == 'ai' for r in records)}")


if __name__ == "__main__":
    main()
