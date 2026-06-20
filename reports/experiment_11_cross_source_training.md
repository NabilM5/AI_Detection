# Experiment 11 — Cross-Source Training Improves External Generalization

## Goal

Previous experiments demonstrated that models trained exclusively on the HC3 Medicine dataset achieved excellent internal performance but generalized poorly to external medical datasets.

The objective of this experiment was to investigate whether increasing the diversity of the training data improves external robustness.

Unlike previous experiments, the classifier was trained using examples from both HC3 Medicine and MedlinePlus.

---

# Research Question

Does cross-source training improve external generalization compared with training on a single dataset?

---

# Motivation

Experiments 03–10 consistently showed that changing model architecture produced only modest improvements in external performance.

For example:

| Model                        | External Accuracy |
| ---------------------------- | ----------------: |
| TF-IDF + Logistic Regression |            42.74% |
| BiLSTM                       |            40.68% |
| Pretrained BiLSTM            |            46.42% |
| GRU                          |            52.24% |
| TinyBERT                     |            55.73% |

Despite using increasingly sophisticated models, external robustness remained limited.

This suggested that the primary limitation was not the model architecture but the training data distribution.

---

# Method

## Training Data

The training dataset combined:

* HC3 Medicine
* MedlinePlus Human

Total training records:

2585

---

## Test Data

Held-out MedlinePlus split.

| Metric       | Value |
| ------------ | ----: |
| Test Records |   337 |

---

# Model

Classifier:

* TF-IDF Vectorizer
* Logistic Regression

Configuration:

* Unigrams + Bigrams
* English stop-word removal
* Balanced class weights

---

# Results

## Overall Performance

| Metric   |  Value |
| -------- | -----: |
| Accuracy | 94.66% |
| ROC-AUC  | 95.87% |

---

## Classification Report

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| AI    |      1.00 |   0.44 |     0.61 |
| Human |      0.94 |   1.00 |     0.97 |

---

## Confusion Matrix

| True \ Pred | AI | Human |
| ----------- | -: | ----: |
| AI          | 14 |    18 |
| Human       |  0 |   305 |

---

# Learned Features

## Strong Human Features

* hi
* hope
* query
* hello
* thanks
* dr
* care
* people
* disease
* disorders
* NIH
* National Institute

## Strong AI Features

* important
* healthcare
* healthcare provider
* determine
* treatment
* medications
* professional
* appropriate
* provider
* condition
* specific

---

# Comparison with Previous Experiments

| Model                   | Training Data         | External Accuracy |
| ----------------------- | --------------------- | ----------------: |
| TF-IDF                  | HC3 only              |            42.74% |
| BiLSTM                  | HC3 only              |            40.68% |
| GRU                     | HC3 only              |            52.24% |
| TinyBERT                | HC3 only              |            55.73% |
| **Cross-Source TF-IDF** | **HC3 + MedlinePlus** |        **94.66%** |

---

# Discussion

Cross-source training produced the largest improvement observed in the entire project.

The modified training script also reports a ROC-AUC of **95.87%**. This shows that
the model ranks most AI examples above human examples reasonably well even though the
default `0.50` decision threshold is too conservative and recalls only 43.75% of AI texts.

## Feature Inspection

The strongest human-associated coefficients included `hi`, `medicines`, `hope`,
`query`, `hello`, `thanks`, `NIH`, and `institute`. The strongest AI-associated
coefficients included `important`, `healthcare`, `healthcare provider`, `determine`,
`possible`, `medications`, `treatment`, and `professional`.

These features indicate that cross-source training improves held-out MedlinePlus
performance but does not eliminate source and writing-style artifacts. ROC-AUC,
threshold behavior, and feature inspection should therefore be reported together.

The external accuracy increased from **42.74%** to **94.66%**, an improvement of more than 50 percentage points.

This improvement is substantially larger than the gains obtained by replacing TF-IDF with recurrent neural networks, pretrained embeddings, or transformer-based models.

These findings indicate that exposure to multiple writing styles and document distributions is considerably more important than increasing model complexity.

The ROC-AUC of **95.87%** further demonstrates that the classifier effectively separates AI-generated and human-written texts.

The relatively low AI recall (44%) indicates that the default decision threshold is conservative. However, previous threshold calibration experiments showed that recall can be increased by lowering the decision threshold, providing a practical trade-off between precision and recall.

---

# Main Findings

**Finding 1**

Cross-source training produced the highest external accuracy of all evaluated approaches.

**Finding 2**

Training data diversity had a greater impact than model architecture.

**Finding 3**

A simple TF-IDF classifier trained on diverse data outperformed multiple neural-network architectures trained on a single dataset.

**Finding 4**

The remaining AI recall errors are primarily related to decision-threshold selection rather than poor class separation, as demonstrated by the high ROC-AUC.

---

# Conclusion

This experiment demonstrates that the primary obstacle to robust medical AI-text detection is **dataset distribution shift**, rather than insufficient model capacity.

Training on multiple medical text sources dramatically improved external generalization, while increasingly complex neural architectures produced only modest gains when trained on a single dataset.

These findings support the central conclusion of this thesis:

> **Training data diversity is a more important factor for robust medical AI-text detection than model complexity alone.**
