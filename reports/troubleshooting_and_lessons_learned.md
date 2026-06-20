# Troubleshooting and Lessons Learned

## Purpose

This document summarizes the major technical problems encountered during the project, how they were diagnosed, and how they were resolved. It also explains the lessons learned from each issue.

---

# 1. TinyBERT Tokenizer Error

## Error

```
ValueError:
Couldn't instantiate the backend tokenizer...
You need to have sentencepiece or tiktoken installed.
```

## Cause

The project initially used:

```python
AutoTokenizer.from_pretrained(...)
```

The environment could not initialize the fast tokenizer because the required backend libraries were missing.

## Solution

The tokenizer implementation was changed to:

```python
BertTokenizer.from_pretrained(...)
```

instead of:

```python
AutoTokenizer.from_pretrained(...)
```

This avoided the dependency on the unavailable fast tokenizer backend.

## Lesson Learned

Transformer models may require additional tokenizer libraries. When compatibility issues occur, using the original tokenizer implementation can be a reliable workaround.

---

# 2. TinyBERT Classification Head Warning

## Warning

```
classifier.weight MISSING
classifier.bias MISSING
```

Several MLM layers were also reported as:

```
UNEXPECTED
```

## Cause

The pretrained checkpoint was originally trained for masked language modeling rather than sequence classification.

The classification layer does not exist in the original checkpoint.

## Solution

The warning was expected.

The classifier layer was initialized randomly and trained during fine-tuning.

No code modification was required.

## Lesson Learned

Not every warning indicates an error. Understanding the pretrained architecture is important before attempting to "fix" warnings.

---

# 3. Missing Vocabulary File

## Error

```
FileNotFoundError:
pretrained_embedding_lstm_vocab.pkl
```

## Cause

The pretrained BiLSTM model was saved, but the vocabulary used during training was not saved.

During evaluation, the tokenizer vocabulary could not be reconstructed correctly.

## Solution

Added:

```python
pickle.dump(vocab, f)
```

after training.

Saved as:

```
models/pretrained_embedding_lstm_vocab.pkl
```

## Lesson Learned

The vocabulary is part of the trained model. Saving only the neural network weights is insufficient for reproducible inference.

---

# 4. Incorrect External Accuracy (18%)

## Problem

The pretrained BiLSTM initially produced an external accuracy of approximately:

```
18%
```

which was inconsistent with all previous experiments.

## Cause

The evaluation script was using a vocabulary different from the one used during training.

As a result, token IDs no longer corresponded to the correct embeddings.

## Solution

The correct vocabulary generated during training was saved and reused during evaluation.

After correction:

```
18%

↓

46%
```

## Lesson Learned

Model weights and vocabulary must always remain synchronized.

---

# 5. Cross-Dataset Performance Collapse

## Observation

HC3 test accuracy:

```
99.74%
```

External MedlinePlus:

```
42.74%
```

Initially this appeared to indicate poor model performance.

## Investigation

Multiple architectures were evaluated:

* LSTM
* GRU
* TinyBERT
* Pretrained BiLSTM

All models showed similar external degradation.

## Solution

Statistical analysis revealed a significant distribution shift between HC3 and MedlinePlus.

Cross-source training substantially improved performance.

## Lesson Learned

Dataset quality and diversity can be more important than increasing model complexity.

---

# 6. Length Bias Hypothesis

## Initial Hypothesis

The model might simply classify longer documents as AI-generated.

## Investigation

A length-only Logistic Regression baseline was trained.

Result:

```
84.97%
```

TF-IDF:

```
99.74%
```

Additional truncation experiments showed:

10 words:

```
94.56%
```

100 words:

```
98.19%
```

Full text:

```
99.74%
```

## Conclusion

Length contributes to prediction, but vocabulary and local context are considerably more informative.

## Lesson Learned

Always verify hypotheses experimentally rather than relying on intuition.

---

# 7. Threshold Selection

## Observation

Using the default probability threshold:

```
0.50
```

resulted in low AI recall.

## Solution

Threshold tuning experiments were performed.

A confidence score was introduced.

A three-class moderation framework was proposed:

* Human
* Uncertain
* AI

## Lesson Learned

Classification thresholds should be selected according to the application rather than using the default value.

---

# 8. Three-Class Moderation

## Problem

Binary classification forces every document into one of two classes.

Borderline cases are often unreliable.

## Solution

Introduced an uncertainty region:

```
score < 0.25
→ Human

0.25–0.55
→ Uncertain

score > 0.55
→ AI
```

Automatic decisions achieved:

```
98.99%
```

while uncertain cases were referred for human review.

## Lesson Learned

Confidence estimation can improve the reliability of practical AI systems.

---

# 9. Statistical Feature Generalization

## Observation

A Logistic Regression classifier using only statistical features achieved:

Internal:

```
93.04%
```

External:

```
35.22%
```

## Cause

The statistical properties of HC3 and MedlinePlus differ significantly.

The classifier learned dataset-specific patterns.

## Lesson Learned

Strong internal performance does not necessarily imply real-world robustness.

---

# 10. Cross-Source Training

## Observation

Training on HC3 only:

```
42.74%
```

Training on HC3 + MedlinePlus:

```
94.66%
```

## Lesson Learned

The largest improvement in the project came from increasing training data diversity rather than replacing the classifier with more complex neural networks.

---

# Key Lessons Learned

1. Always verify assumptions through experiments.

2. Compare models on external datasets, not only internal test sets.

3. Save every artifact required for inference (model, vocabulary, tokenizer, configuration).

4. Dataset distribution has a larger impact on generalization than model architecture.

5. Confidence estimation and human-in-the-loop moderation are practical strategies for deploying AI detection systems in real-world healthcare applications.

6. Negative results are scientifically valuable. Discovering why a model fails often provides more insight than simply reporting high accuracy.

