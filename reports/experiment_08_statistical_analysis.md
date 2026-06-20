# Experiment 08 — Statistical Analysis of Human and AI Medical Texts

## Goal

The objective of this experiment was to investigate whether human-written and AI-generated medical texts exhibit measurable statistical differences independent of machine learning models.

Unlike previous experiments, this analysis does not rely on classifiers, embeddings, or neural networks.

The purpose is to identify intrinsic properties of the datasets and explain the behavior of AI-detection models.

---

# Research Question

Do human and AI medical texts differ significantly in their statistical characteristics?

---

# Datasets

Four groups were analyzed:

| Dataset               | Label | Documents |
| --------------------- | ----: | --------: |
| HC3 Medicine          | Human |      1248 |
| HC3 Medicine          |    AI |      1334 |
| MedlinePlus           | Human |      1016 |
| MedlinePlus Generated |    AI |       100 |

---

# Extracted Features

The following statistical properties were computed for every document:

* Word Count
* Sentence Count
* Average Sentence Length
* Lexical Diversity (Type-Token Ratio)
* Average Word Length
* Punctuation Density

---

# Statistical Summary

## HC3 Medicine

| Feature             |  Human |     AI |
| ------------------- | -----: | -----: |
| Word Count          |  85.90 | 189.69 |
| Sentence Count      |   7.95 |   8.89 |
| Avg Sentence Length |  12.89 |  21.69 |
| Lexical Diversity   | 0.7769 | 0.5532 |
| Avg Word Length     |   4.74 |   4.88 |
| Punctuation Density | 0.1649 | 0.1241 |

---

## MedlinePlus

| Feature             |  Human |     AI |
| ------------------- | -----: | -----: |
| Word Count          | 341.51 | 171.72 |
| Sentence Count      |  23.61 |  10.94 |
| Avg Sentence Length |  14.86 |  16.29 |
| Lexical Diversity   | 0.5559 | 0.7358 |
| Avg Word Length     |   4.81 |   5.47 |
| Punctuation Density | 0.1519 | 0.2054 |

---

# Statistical Significance Testing

Welch's t-test was applied to compare distributions.

Results showed highly significant differences across almost all comparisons.

Examples:

## HC3 Human vs HC3 AI

| Feature             | P-value |
| ------------------- | ------: |
| Word Count          | < 0.001 |
| Avg Sentence Length | < 0.001 |
| Lexical Diversity   | < 0.001 |

## MedlinePlus Human vs MedlinePlus AI

| Feature             |  P-value |
| ------------------- | -------: |
| Word Count          |  < 0.001 |
| Avg Sentence Length | 0.000568 |
| Lexical Diversity   |  < 0.001 |

---

# Distribution Shift

The most important finding concerns lexical diversity.

| Group             | Lexical Diversity |
| ----------------- | ----------------: |
| HC3 Human         |            0.7769 |
| HC3 AI            |            0.5532 |
| MedlinePlus Human |            0.5559 |
| MedlinePlus AI    |            0.7358 |

A near-complete reversal is observed.

HC3:

Human > AI

MedlinePlus:

AI > Human

---

# Interpretation

This distribution reversal explains why models trained on HC3 generalized poorly to MedlinePlus.

HC3-trained models learn:

* High lexical diversity → Human
* Low lexical diversity → AI

However, MedlinePlus exhibits the opposite pattern.

Consequently, external human texts are frequently misclassified as AI-generated.

---

# Main Findings

Finding 1:

Human and AI medical texts differ significantly in multiple statistical properties.

Finding 2:

Lexical diversity is the strongest discriminative feature.

Finding 3:

The relationship between lexical diversity and class label changes across datasets.

Finding 4:

Dataset distribution shift is a major factor affecting external generalization.

---

# Conclusion

Statistical analysis revealed substantial differences between HC3 and MedlinePlus distributions. These differences are highly significant and provide a strong explanation for the failure of HC3-trained models on external medical datasets.

The results suggest that dataset characteristics may influence detection performance more strongly than model architecture.

