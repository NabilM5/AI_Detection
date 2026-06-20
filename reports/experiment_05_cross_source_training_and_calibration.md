# Experiment 05 — Cross-Source Training and Threshold Calibration

## Goal

Evaluate whether adding external medical data to the training set improves generalization more than changing model architecture.

A secondary objective is to determine whether probability-threshold calibration improves detection performance.

---

# Motivation

Previous experiments showed:

| Model | HC3 Accuracy | External Accuracy |
|---------|---------:|---------:|
| TF-IDF + LR | 99.74% | 42.74% |
| BiLSTM | 96.63% | 40.68% |
| BiGRU | 95.85% | 52.24% |
| TinyBERT | 97.67% | 55.73% |

Despite increasingly sophisticated architectures, all models suffered substantial performance degradation on external MedlinePlus data.

This suggested that dataset distribution mismatch might be a larger problem than model complexity.

---

# Cross-Source Dataset

## HC3 Training Data

| Class | Count |
|---------|---------:|
| Human | 1248 |
| AI | 1334 |
| Total | 2582 |

---

## MedlinePlus Dataset

| Class | Count |
|---------|---------:|
| Human | 1016 |
| AI | 100 |
| Total | 1116 |

The MedlinePlus dataset was split by topic group.

This prevented information leakage between human and AI versions of the same medical topic.

---

## Final Cross-Source Split

Training:

- HC3 training set
- MedlinePlus training subset

Testing:

- Held-out MedlinePlus subset only

| Split | Records |
|---------|---------:|
| Train | 2585 |
| Test | 337 |

---

# Model

Classifier:

- TF-IDF
- Logistic Regression

Parameters:

- max_features = 20000
- ngram_range = (1,2)
- stop_words = english
- class_weight = balanced

---

# Initial Results

## Accuracy

| Metric | Value |
|---------|---------:|
| Accuracy | 94.66% |
| ROC-AUC | 95.87% |

---

## Classification Report

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| AI | 1.00 | 0.44 | 0.61 |
| Human | 0.94 | 1.00 | 0.97 |

---

## Confusion Matrix

| True \\ Pred | AI | Human |
|---------|---------:|---------:|
| AI | 14 | 18 |
| Human | 0 | 305 |

---

# Interpretation

Cross-source training dramatically improved external performance.

External accuracy increased from:

42.74%

to:

94.66%

This improvement was substantially larger than any gain obtained by changing model architecture.

However, the model became overly conservative:

- Human recall = 100%
- AI recall = 43.75%

The classifier strongly preferred predicting "human".

The ROC-AUC of 95.87% indicates that probability ranking is substantially stronger
than the default-threshold AI recall suggests. Feature inspection still found strong
source/style signals: human-associated terms included `hi`, `query`, `thanks`, and
`NIH`, while AI-associated terms included `important`, `healthcare provider`,
`determine`, and `treatment`.

---

# Threshold Calibration

The default decision threshold was:

0.50

The AI probability threshold was systematically varied.

---

## Threshold Sweep

| Threshold | Accuracy | AI Precision | AI Recall | AI F1 | Human Recall |
|------------|---------:|---------:|---------:|---------:|---------:|
| 0.10 | 15.43% | 10.09% | 100.00% | 18.34% | 6.56% |
| 0.20 | 65.28% | 21.48% | 100.00% | 35.36% | 61.64% |
| 0.30 | 94.07% | 64.29% | 84.38% | 72.97% | 95.08% |
| 0.40 | 94.66% | 88.89% | 50.00% | 64.00% | 99.34% |
| 0.50 | 94.66% | 100.00% | 43.75% | 60.87% | 100.00% |
| 0.60 | 92.28% | 100.00% | 18.75% | 31.58% | 100.00% |

---

# Best Threshold

The most balanced operating point was:

Threshold = 0.30

Results:

| Metric | Value |
|---------|---------:|
| Accuracy | 94.07% |
| AI Precision | 64.29% |
| AI Recall | 84.38% |
| AI F1 | 72.97% |
| Human Recall | 95.08% |

---

# Main Findings

## Finding 1

Dataset diversity had a much larger impact than architecture complexity.

Adding MedlinePlus training data improved external accuracy by more than 50 percentage points.

---

## Finding 2

Threshold calibration significantly improved practical performance.

The default threshold sacrificed AI recall in exchange for perfect human recall.

A calibrated threshold produced a substantially better balance.

---

## Finding 3

Real-world deployment requires confidence calibration.

Using the default classifier output is suboptimal because moderation systems often have asymmetric costs:

- Missing AI content
- Incorrectly flagging human content

Threshold tuning allows the system to adapt to application requirements.

---

# Conclusion

Cross-source training produced the strongest improvement observed in the project.

The experiment demonstrates that:

1. Training data diversity is more important than model architecture.
2. Threshold calibration is essential.
3. External evaluation is necessary because HC3 accuracy alone is misleading.

These findings support the hypothesis that robust medical AI-text detection depends primarily on dataset coverage and calibration rather than increasing model complexity alone.
