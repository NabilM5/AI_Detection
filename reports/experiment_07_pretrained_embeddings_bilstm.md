# Experiment 07 — Pretrained Embeddings + BiLSTM

## Goal

The supervisor suggested investigating:

> fastText + LSTM

The objective of this experiment was to determine whether pretrained semantic word embeddings improve AI-generated medical text detection compared with randomly initialized embeddings.

The hypothesis was that pretrained embeddings provide richer semantic information and therefore improve generalization to unseen medical texts.

---

# Research Question

Can pretrained word embeddings improve external robustness compared with a standard BiLSTM trained from scratch?

---

# Motivation

Previous experiments demonstrated a significant gap between:

* Internal HC3 performance
* External MedlinePlus performance

Although recurrent neural networks achieved high accuracy on HC3, they generalized poorly to external medical datasets.

A possible explanation is that embeddings learned only from HC3 do not capture broader medical language semantics.

Pretrained embeddings provide prior linguistic knowledge learned from large corpora and may improve transferability.

---

# Method

## Baseline Model

Architecture:

* Embedding Layer
* Bidirectional LSTM
* Dropout
* Linear Classification Layer

Embedding initialization:

Random

---

## Pretrained Embedding Model

Architecture:

* Pretrained GloVe Embedding Layer
* Bidirectional LSTM
* Dropout
* Linear Classification Layer

Embedding initialization:

Pretrained

---

# Embeddings

Model:

GloVe Wiki Gigaword 100

Properties:

| Property        |                Value |
| --------------- | -------------------: |
| Dimension       |                  100 |
| Vocabulary Size |              400,000 |
| Training Corpus | Wikipedia + Gigaword |

Embedding coverage on HC3 vocabulary:

| Metric          |  Value |
| --------------- | -----: |
| Coverage        | 42.80% |
| Vocabulary Size | 16,607 |
| Covered Tokens  |  7,108 |

---

# Dataset

Training:

HC3 Medicine

Testing:

HC3 Medicine Test Split

| Split | Records |
| ----- | ------: |
| Train |   1,806 |
| Test  |     386 |

---

# Internal Evaluation

## Pretrained Embedding + BiLSTM

| Metric   |  Value |
| -------- | -----: |
| Accuracy | 97.67% |

### Classification Report

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| Human |      0.98 |   0.97 | 0.98 |
| AI    |      0.97 |   0.98 | 0.98 |

---

# Comparison with Random Embeddings

| Model                          | Accuracy |
| ------------------------------ | -------: |
| BiLSTM (Random Embeddings)     |   96.63% |
| BiLSTM (Pretrained Embeddings) |   97.67% |

Improvement:

+1.04 percentage points

---

# External Evaluation

Dataset:

MedlinePlus Human + Generated AI

| Metric        | Value |
| ------------- | ----: |
| Human Records |  1016 |
| AI Records    |   100 |
| Total         |  1116 |

---

## Results

| Metric   |  Value |
| -------- | -----: |
| Accuracy | 46.42% |

### Classification Report

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| Human |      0.93 |   0.44 | 0.60 |
| AI    |      0.11 |   0.67 | 0.18 |

---

### Confusion Matrix

| True \ Pred | Human |  AI |
| ----------- | ----: | --: |
| Human       |   451 | 565 |
| AI          |    33 |  67 |

---

# Comparison with Previous Models

| Model                         | External Accuracy |
| ----------------------------- | ----------------: |
| TF-IDF + Logistic Regression  |            42.74% |
| BiLSTM                        |            40.68% |
| Pretrained Embedding + BiLSTM |            46.42% |
| BiGRU                         |            52.24% |
| TinyBERT                      |            55.73% |

---

# Analysis

The pretrained embeddings improved both:

* Internal performance
* External performance

However, the improvement was relatively small.

External accuracy increased from:

40.68%

to:

46.42%

This indicates that pretrained semantic information helps the model generalize beyond HC3.

Nevertheless, the model still suffers from severe domain-shift problems.

The external improvement is much smaller than the gains obtained through cross-source training.

---

# Main Findings

Finding 1:

Pretrained embeddings improved HC3 accuracy.

Finding 2:

Pretrained embeddings improved external accuracy.

Finding 3:

The improvement was modest.

Finding 4:

Dataset diversity had a larger impact than embedding initialization.

---

# Conclusion

The experiment demonstrates that pretrained embeddings provide measurable benefits for AI-generated medical text detection.

However, the results also show that semantic embeddings alone are insufficient to overcome dataset distribution mismatch.

Compared with architecture modifications, cross-source training remains the most effective strategy for improving external robustness.

These findings support the broader conclusion of the project:

Training data diversity contributes more to generalization than model complexity or embedding initialization alone.

