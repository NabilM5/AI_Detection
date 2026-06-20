import json
from pathlib import Path

from datasets import load_dataset

DATASET_PATH = (
    "hf://datasets/Hello-SimpleAI/HC3@refs/convert/parquet/medicine/train/*.parquet"
)
OUTPUT_PATH = Path("data/raw/hc3_medicine_raw.json")


def main():
    print("Downloading the English HC3 medical dataset...")
    dataset = load_dataset(
        "parquet",
        data_files=DATASET_PATH,
        split="train",
    )

    records = [
        {
            "question": item.get("question"),
            "human_medical_answers": item.get("human_answers"),
            "chatgpt_medical_answers": item.get("chatgpt_answers"),
        }
        for item in dataset
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} medical question-answer pairs to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
