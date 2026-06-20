import json
import pickle

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader


MODEL_PATH = "models/pretrained_embedding_lstm.pt"
VOCAB_PATH = "models/pretrained_embedding_lstm_vocab.pkl"

HUMAN_PATH = "data/processed/medlineplus_human.jsonl"
AI_PATH = "data/processed/medlineplus_ai.jsonl"

MAX_LEN = 200
EMBED_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def tokenize(text):
    return text.lower().split()


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
        self.label_map = {"human": 0, "ai": 1}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        x = encode_text(record["text"], self.vocab)
        y = self.label_map[record["label"]]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(HIDDEN_DIM * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        combined = self.dropout(combined)

        return self.classifier(combined)


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

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    human_records = load_jsonl(HUMAN_PATH)
    ai_records = load_jsonl(AI_PATH)
    records = human_records + ai_records

    dataset = TextDataset(records, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    y_true, y_pred = evaluate(model, dataloader)

    print("\nPretrained Embedding + BiLSTM MedlinePlus External Evaluation")
    print("=" * 50)
    print(f"Human records: {len(human_records)}")
    print(f"AI records:    {len(ai_records)}")
    print(f"Total:         {len(records)}")

    print("\nAccuracy")
    print("=" * 50)
    print(f"{accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification Report")
    print("=" * 50)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["human", "ai"],
        )
    )

    print("\nConfusion Matrix")
    print("=" * 50)
    print("Labels: [human, ai]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
