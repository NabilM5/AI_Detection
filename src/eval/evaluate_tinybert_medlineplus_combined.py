import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = "models/tinybert_baseline"

HUMAN_PATH = "data/processed/medlineplus_human.jsonl"
AI_PATH = "data/processed/medlineplus_ai.jsonl"

MAX_LEN = 256
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


class TextDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
        self.label_map = {"human": 0, "ai": 1}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        encoded = self.tokenizer(
            record["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[record["label"]], dtype=torch.long),
        }


def evaluate(model, dataloader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


def main():
    print(f"Using device: {DEVICE}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    human_records = load_jsonl(HUMAN_PATH)
    ai_records = load_jsonl(AI_PATH)
    records = human_records + ai_records

    dataset = TextDataset(records, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = evaluate(model, dataloader)

    print("\nTinyBERT Combined MedlinePlus External Evaluation")
    print("=" * 50)
    print(f"Human records: {len(human_records)}")
    print(f"AI records:    {len(ai_records)}")
    print(f"Total:         {len(records)}")

    print("\nAccuracy")
    print("=" * 50)
    print(f"{accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["human", "ai"]))

    print("\nConfusion Matrix")
    print("=" * 50)
    print("Labels: [human, ai]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
