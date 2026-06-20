# Experiment 06 — AI Confidence Score and Three-Class Moderation Framework

## Goal

Previous experiments focused on binary classification:

* Human
* AI

However, real-world medical moderation systems often require a more nuanced decision process.

The goal of this experiment was to investigate whether a confidence-based framework could improve reliability by introducing a third category:

* Human
* Uncertain
* AI

This experiment directly addresses the supervisor's recommendation to explore confidence estimation and soft-label approaches.

---

# Motivation

A binary classifier is forced to make a decision even when confidence is low.

This can lead to:

* False positives (human text classified as AI)
* False negatives (AI text classified as human)

In healthcare-related environments, both error types can be costly.

Instead of forcing a decision, uncertain cases can be escalated for human review.

---

# AI Confidence Score

The confidence score combines:

1. Model AI probability
2. Text length
3. Lexical diversity
4. Medical terminology density

Version 2 formula:

Score =

0.80 × AI_Probability

* 0.10 × Length_Score

* 0.05 × Lexical_Diversity

* 0.05 × Medical_Density

Where:

* AI_Probability comes from the calibrated cross-source TF-IDF classifier.
* Length_Score normalizes document length.
* Lexical_Diversity measures vocabulary richness.
* Medical_Density estimates the concentration of medical terminology.

---

# Three-Class Decision Rules

Human:

Score < 0.25

Uncertain:

0.25 ≤ Score < 0.55

AI:

Score ≥ 0.55

---

# Dataset

Evaluation dataset:

Cross-source held-out MedlinePlus test set.

| Metric  | Value |
| ------- | ----: |
| Records |   337 |

---

# Prediction Distribution

| Class     | Count |
| --------- | ----: |
| Human     |   191 |
| Uncertain |   139 |
| AI        |     7 |

---

# Human Review Rate

The system automatically classified:

198 records

and sent:

139 records

for manual review.

Review percentage:

41.25%

---

# Auto-Decided Evaluation

Only records classified directly as Human or AI were evaluated.

| Metric               |  Value |
| -------------------- | -----: |
| Auto-decided records |    198 |
| Accuracy             | 98.99% |

---

# Classification Report

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| AI    |      1.00 |   0.78 | 0.88 |
| Human |      0.99 |   1.00 | 0.99 |

---

# Confusion Matrix

| True \ Pred | AI | Human |
| ----------- | -: | ----: |
| AI          |  7 |     2 |
| Human       |  0 |   189 |

---

# Discussion

The confidence framework substantially improved reliability on automatically decided cases.

Instead of forcing predictions on borderline examples, the system deferred uncertain texts for human review.

This reduced the number of incorrect automatic decisions while preserving high confidence in accepted classifications.

The experiment demonstrates that confidence estimation can be more valuable than increasing model complexity.

---

# Real-Time User Scenario

Consider a hospital communication platform receiving:

* Patient questions
* Educational content
* AI-assisted responses
* Staff-generated messages

The moderation system operates as follows:

1. Analyze incoming text.
2. Compute AI confidence score.
3. Assign one of three outcomes:

Human:
Published immediately.

AI:
Flagged automatically.

Uncertain:
Sent to a moderator for review.

This workflow allows human experts to focus only on difficult cases.

---

# Main Findings

Finding 1:

A confidence-based framework is more practical than a binary classifier.

Finding 2:

The system achieved 98.99% accuracy on automatically decided cases.

Finding 3:

Human review is required for approximately 41% of cases.

Finding 4:

Confidence estimation directly supports real-time moderation and risk management.

---

# Conclusion

The proposed three-class moderation framework demonstrates that AI-generated medical text detection should not be treated purely as a binary classification task.

By introducing an uncertainty region and confidence-based decision making, the system achieved extremely high accuracy on automatically processed records while safely escalating ambiguous cases for human review.

This approach better reflects real-world healthcare moderation requirements and provides a foundation for future work on soft labels, confidence calibration, and human-in-the-loop AI systems.

