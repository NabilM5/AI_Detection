import json
import os
import pickle
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/splits/train.jsonl"
TEST_PATH = "data/processed/splits/test.jsonl"

MAX_VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def tokenize(text):
    return text.lower().split()


def build_vocab(records):
    counter = Counter()

    for record in records:
        counter.update(tokenize(record["text"]))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

    return ids


class TextDataset(Dataset):
    def __init__(self, records, vocab):
        self.records = records
        self.vocab = vocab
        self.label_map = {
            "human": 0,
            "ai": 1,
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        x = encode_text(record["text"], self.vocab)
        y = self.label_map[record["label"]]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=0,
        )

        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(HIDDEN_DIM * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)

        _, hidden = self.gru(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        combined = self.dropout(combined)

        return self.classifier(combined)


def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)

            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    return all_labels, all_preds


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs("models", exist_ok=True)

    train_records = load_jsonl(TRAIN_PATH)
    test_records = load_jsonl(TEST_PATH)

    vocab = build_vocab(train_records)

    train_dataset = TextDataset(train_records, vocab)
    test_dataset = TextDataset(test_records, vocab)

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

    model = GRUClassifier(vocab_size=len(vocab)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f}")

    y_true, y_pred = evaluate(model, test_loader)

    print("\nGRU Baseline")
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

    torch.save(model.state_dict(), "models/gru_baseline.pt")
    print("\nSaved model to models/gru_baseline.pt")

    with open("models/gru_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Saved vocabulary to models/gru_vocab.pkl")


if __name__ == "__main__":
    main()
