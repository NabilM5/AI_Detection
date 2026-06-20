# Project Timeline

## Project Title

Detection of AI-Generated Medical Text

---

# Research Objective

Develop a system capable of distinguishing AI-generated medical text from human-written medical text while evaluating its robustness on external datasets.

---

# Initial Hypothesis

The project originally assumed that modern NLP models would generalize well from HC3 Medicine to other medical datasets.

The experiments gradually demonstrated that this assumption was incorrect.

The primary challenge proved to be dataset distribution shift rather than model architecture.

---

# Experiment 01 — Dataset Exploration

## Goal

Understand HC3 Medicine.

## Tasks

* Load HC3 Medicine
* Count records
* Analyze labels
* Explore examples

## Main Finding

Human and AI responses differ considerably in writing style and length.

---

# Experiment 02 — TF-IDF + Logistic Regression

## Goal

Build a classical baseline.

## Result

Internal Accuracy:

99.74%

## Finding

Very high internal performance.

---

# Experiment 03 — External Evaluation

## Goal

Evaluate robustness using MedlinePlus.

## Result

Accuracy:

42.74%

## Finding

Performance collapsed.

The model failed to generalize.

---

# Experiment 04 — Length Analysis

## Goal

Determine whether document length explains prediction.

## Experiments

* Length-only Logistic Regression
* Length correlation
* Length buckets

## Result

Length-only:

84.97%

## Finding

Length contributes substantially but does not fully explain performance.

---

# Experiment 05 — TF-IDF Analysis

## Goal

Understand what TF-IDF learned.

## Experiments

* Top features
* Truncation study

## Findings

The first 100 words already contain most predictive information.

Vocabulary contributes more than document length.

---

# Experiment 06 — Neural Baselines

## Models

* BiLSTM
* BiGRU
* TinyBERT
* Pretrained BiLSTM

## Internal Results

BiLSTM:

96.63%

GRU:

95.85%

TinyBERT:

97.67%

Pretrained BiLSTM:

97.67%

## External Results

All models experienced substantial performance degradation.

## Finding

Increasing model complexity alone did not solve the problem.

---

# Experiment 07 — Statistical Analysis

## Goal

Analyze datasets independently of machine learning models.

## Features

* Word count
* Sentence length
* Lexical diversity
* Average word length
* Punctuation density

## Finding

Significant statistical differences exist between datasets.

---

# Experiment 08 — Statistical Significance Testing

## Method

Welch's t-test

## Finding

Nearly all feature differences were statistically significant.

Most importantly:

HC3 Human

↓

High lexical diversity

MedlinePlus Human

↓

Low lexical diversity

This reversal explains poor external performance.

---

# Experiment 09 — Distribution Visualization

## Goal

Visualize dataset shift.

## Figures

* Word Count
* Lexical Diversity
* Sentence Length

## Finding

Distribution shift is immediately visible.

---

# Experiment 10 — Statistical Feature Classifier

## Goal

Evaluate whether statistical properties alone can distinguish AI text.

## Internal

93.04%

## External

35.22%

## Finding

Simple statistical features explain much of HC3 performance but generalize poorly.

---

# Experiment 11 — Confidence-Based Moderation

## Goal

Introduce uncertainty-aware prediction.

## Classes

* Human
* Uncertain
* AI

## Finding

Automatic decisions achieved:

98.99%

while uncertain cases were forwarded for manual review.

---

# Experiment 12 — Cross-Source Training

## Goal

Improve external robustness by increasing dataset diversity.

## Training

HC3 + MedlinePlus

## External Result

94.66%

ROC-AUC:

95.87%

## Finding

This produced the largest improvement in the entire project.

Training data diversity had a much larger effect than changing model architecture.

---

# Final Research Conclusions

## Conclusion 1

HC3 Medicine contains strong stylistic patterns that are easily learned.

---

## Conclusion 2

Internal accuracy alone is not an indicator of real-world performance.

---

## Conclusion 3

Dataset distribution shift is the primary cause of external performance degradation.

---

## Conclusion 4

Increasing model complexity produces relatively small improvements compared with increasing dataset diversity.

---

## Conclusion 5

Confidence estimation enables practical human-in-the-loop moderation.

---

## Conclusion 6

Cross-source training is the most effective strategy evaluated in this project.

---

# Main Scientific Contribution

This work demonstrates that the principal challenge in medical AI-text detection is not insufficient model capacity but distribution shift between medical text sources.

The experiments show that increasing training data diversity dramatically improves external robustness, while confidence-aware moderation provides a practical deployment strategy for real-world healthcare systems.

---

# Personal Lessons Learned

During this project I learned to:

* Design reproducible machine learning experiments.
* Build and compare classical and neural NLP models.
* Perform statistical analysis of text.
* Evaluate models on external datasets.
* Investigate failure cases rather than relying only on accuracy.
* Use confidence estimation for safer AI deployment.
* Explain model behavior through data analysis rather than treating models as black boxes.

