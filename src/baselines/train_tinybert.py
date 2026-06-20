import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/splits/train.jsonl"
TEST_PATH = "data/processed/splits/test.jsonl"

MODEL_NAME = "prajjwal1/bert-tiny"
SAVE_DIR = "models/tinybert_baseline"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

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
        self.label_map = {
            "human": 0,
            "ai": 1,
        }

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

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[record["label"]], dtype=torch.long),
        }

        return item


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


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
    print(f"Model: {MODEL_NAME}")

    train_records = load_jsonl(TRAIN_PATH)
    test_records = load_jsonl(TEST_PATH)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextDataset(train_records, tokenizer)
    test_dataset = TextDataset(test_records, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    config = BertConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f}")

    y_true, y_pred = evaluate(model, test_loader)

    print("\nTinyBERT Baseline")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification Report")
    print("=" * 50)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["human", "ai"],
        )
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"\nSaved TinyBERT model to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
