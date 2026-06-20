# Experiment 10 — Statistical Feature Classifier

## Goal

The objective of this experiment was to determine how much AI-text detection performance can be achieved using only simple statistical features.

No lexical features, embeddings, or language models were used.

---

# Research Question

Can human and AI medical texts be distinguished using only statistical properties of the text?

---

# Features

The classifier used:

* Word Count
* Average Sentence Length
* Lexical Diversity
* Average Word Length
* Punctuation Density

Classifier:

Logistic Regression

---

# HC3 Internal Evaluation

## Results

| Metric   |  Value |
| -------- | -----: |
| Accuracy | 93.04% |

### Classification Report

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| AI    |      0.91 |   0.96 | 0.93 |
| Human |      0.96 |   0.89 | 0.93 |

---

# Feature Importance

| Feature             | Coefficient |
| ------------------- | ----------: |
| Lexical Diversity   |      8.1625 |
| Punctuation Density |      1.2278 |
| Avg Word Length     |     -1.1286 |
| Avg Sentence Length |     -0.0903 |
| Word Count          |     -0.0153 |

Lexical diversity was by far the most influential feature.

---

# External Evaluation

Dataset:

MedlinePlus Human + Generated AI

| Metric   |  Value |
| -------- | -----: |
| Records  |   1116 |
| Accuracy | 35.22% |

---

# Classification Report

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| AI    |      0.09 |   0.64 | 0.15 |
| Human |      0.90 |   0.32 | 0.48 |

---

# Confusion Matrix

| True \ Pred |  AI | Human |
| ----------- | --: | ----: |
| AI          |  64 |    36 |
| Human       | 687 |   329 |

---

# Comparison with Other Models

| Model                | External Accuracy |
| -------------------- | ----------------: |
| Statistical Features |            35.22% |
| TF-IDF               |            42.74% |
| BiLSTM               |            40.68% |
| Pretrained BiLSTM    |            46.42% |
| GRU                  |            52.24% |
| TinyBERT             |            55.73% |

---

# Interpretation

The statistical classifier achieved strong performance on HC3 despite using only five simple features.

This indicates that HC3 contains strong stylistic and statistical artifacts that distinguish human and AI texts.

However, performance collapsed on MedlinePlus because these statistical patterns do not generalize across datasets.

The classifier learned dataset-specific correlations rather than universal characteristics of AI-generated medical text.

---

# Main Findings

Finding 1:

Statistical features alone achieved 93.04% accuracy on HC3.

Finding 2:

Lexical diversity was the dominant predictive feature.

Finding 3:

External performance dropped to 35.22%.

Finding 4:

Dataset shift has a larger impact than feature complexity.

---

# Conclusion

The experiment demonstrates that much of the HC3 detection performance can be explained by simple statistical properties rather than deep semantic understanding.

At the same time, the poor external performance confirms that these properties are strongly dataset-dependent and are insufficient for robust real-world AI-text detection.

