# Experiment 03 — Neural Baselines (BiLSTM vs BiGRU)

## Goal

Evaluate whether sequence-based neural models improve generalization compared with TF-IDF.

The hypothesis was that recurrent neural networks could better capture local linguistic context and therefore be less sensitive to HC3-specific stylistic artifacts.

---

# Models

## BiLSTM

Architecture:

- Trainable word embeddings
- Bidirectional LSTM
- Hidden size: 128
- Dropout: 0.3
- Fully connected classifier

Training:

- Epochs: 5
- Optimizer: Adam
- Learning rate: 1e-3

---

## BiGRU

Architecture:

- Trainable word embeddings
- Bidirectional GRU
- Hidden size: 128
- Dropout: 0.3
- Fully connected classifier

Training:

- Epochs: 5
- Optimizer: Adam
- Learning rate: 1e-3

---

# HC3 Internal Evaluation

## BiLSTM

| Metric | Value |
|---------|---------:|
| Accuracy | 96.63% |

### Classification

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Human | 0.99 | 0.94 | 0.96 |
| AI | 0.95 | 0.99 | 0.97 |

---

## BiGRU

| Metric | Value |
|---------|---------:|
| Accuracy | 95.85% |

### Classification

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Human | 0.93 | 0.98 | 0.96 |
| AI | 0.98 | 0.93 | 0.96 |

---

# External MedlinePlus Benchmark

Dataset:

| Class | Count |
|---------|---------:|
| Human | 1016 |
| AI | 100 |
| Total | 1116 |

---

# BiLSTM External Results

| Metric | Value |
|---------|---------:|
| Accuracy | 40.68% |

### Classification

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Human | 0.93 | 0.38 | 0.54 |
| AI | 0.10 | 0.73 | 0.18 |

### Confusion Matrix

| True \\ Pred | Human | AI |
|---------|---------:|---------:|
| Human | 381 | 635 |
| AI | 27 | 73 |

---

# BiGRU External Results

| Metric | Value |
|---------|---------:|
| Accuracy | 52.24% |

### Classification

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Human | 0.92 | 0.52 | 0.66 |
| AI | 0.10 | 0.54 | 0.17 |

### Confusion Matrix

| True \\ Pred | Human | AI |
|---------|---------:|---------:|
| Human | 529 | 487 |
| AI | 46 | 54 |

---

# Comparison

| Model | HC3 Accuracy | External Accuracy |
|---------|---------:|---------:|
| TF-IDF + Logistic Regression | 99.74% | 42.74% |
| BiLSTM | 96.63% | 40.68% |
| BiGRU | 95.85% | 52.24% |

---

# Discussion

The recurrent neural models achieved high accuracy on HC3, but none generalized well to the external MedlinePlus benchmark.

BiLSTM and TF-IDF produced very similar failure modes:

- High AI recall
- Extremely poor human recall
- Large numbers of false positives

BiGRU produced a more balanced prediction distribution and achieved the highest external accuracy among the tested models, but performance remained inadequate for real-world deployment.

The results suggest that model architecture is not the primary bottleneck.

Instead, the main limitation appears to be dataset distribution mismatch between HC3 and external medical writing sources.

---

# Main Finding

Replacing TF-IDF with recurrent neural networks did not solve the generalization problem.

All models trained on HC3 experienced substantial performance degradation on external medical texts.

This indicates that robust AI-generated medical text detection requires more diverse training data rather than simply more complex architectures.

---

# Next Stage

Following supervisor recommendations, the next experiments will include:

1. Pretrained fastText embeddings.
2. TinyBERT baseline.
3. Multi-source training.
4. Confidence estimation and calibration.
5. Statistical and topological text analysis.
