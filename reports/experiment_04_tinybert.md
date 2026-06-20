# Experiment 04 — TinyBERT Baseline

## Goal

Evaluate whether a transformer-based architecture improves generalization compared with:

- TF-IDF + Logistic Regression
- BiLSTM
- BiGRU

The hypothesis was that contextual transformer representations would be more robust to stylistic differences between HC3 and external medical datasets.

---

# Model

Base model:

- prajjwal1/bert-tiny

Architecture:

- Transformer encoder
- Sequence classification head
- Two output classes:
  - human
  - ai

Training parameters:

| Parameter | Value |
|------------|--------:|
| Max length | 256 |
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |

---

# Training Dataset

HC3 Medicine

| Class | Count |
|---------|---------:|
| Human | 1248 |
| AI | 1334 |
| Total | 2582 |

Grouped train/test split was used to prevent leakage between human and AI answers originating from the same question.

---

# HC3 Internal Evaluation

## Accuracy

| Metric | Value |
|---------|---------:|
| Accuracy | 97.67% |

---

## Classification Report

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Human | 0.98 | 0.97 | 0.98 |
| AI | 0.97 | 0.98 | 0.98 |

---

# External Benchmark

Dataset:

MedlinePlus Human + MedlinePlus AI

| Class | Count |
|---------|---------:|
| Human | 1016 |
| AI | 100 |
| Total | 1116 |

---

# External Evaluation

## Accuracy

| Metric | Value |
|---------|---------:|
| Accuracy | 55.73% |

---

## Classification Report

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Human | 0.93 | 0.55 | 0.70 |
| AI | 0.12 | 0.59 | 0.19 |

---

## Confusion Matrix

Rows = True Labels

Columns = Predictions

| True \\ Pred | Human | AI |
|---------|---------:|---------:|
| Human | 563 | 453 |
| AI | 41 | 59 |

---

# Comparison With Previous Models

| Model | HC3 Accuracy | External Accuracy |
|---------|---------:|---------:|
| TF-IDF + Logistic Regression | 99.74% | 42.74% |
| BiLSTM | 96.63% | 40.68% |
| BiGRU | 95.85% | 52.24% |
| TinyBERT | 97.67% | 55.73% |

---

# Discussion

TinyBERT produced the best external benchmark result among all tested models.

Compared with recurrent neural networks, TinyBERT was more robust to stylistic variation between HC3 and MedlinePlus.

However, the model still produced a large number of false positives:

- 453 human MedlinePlus articles were classified as AI-generated.

Although transformer representations improved performance, the improvement was moderate rather than transformative.

---

# Main Finding

The strongest tested architecture still failed to generalize reliably outside HC3.

This suggests that model architecture alone is not sufficient for robust medical AI-text detection.

Dataset diversity and cross-source training appear to be more important factors than classifier complexity.

---

# Conclusion

TinyBERT achieved:

- Strong HC3 performance (97.67%)
- Best external performance (55.73%)

Despite outperforming TF-IDF, LSTM, and GRU on external evaluation, the model remained far below the reliability required for deployment.

The results support the hypothesis that current performance limitations are driven primarily by dataset distribution mismatch rather than model capacity.

Future work should focus on:

1. Multi-source training datasets.
2. fastText pretrained embeddings.
3. Confidence calibration.
4. Statistical and topological text features.
5. Soft-label approaches for AI-generation confidence estimation.
