# Experiment 02 — External MedlinePlus Benchmark

## Goal

Evaluate whether the HC3-trained AI detector generalizes to a different medical text source.

The model was trained only on the HC3 Medicine dataset and tested on MedlinePlus patient-education articles and AI-generated MedlinePlus articles.

---

# Training Data

Dataset:

- HC3 Medicine

Training model:

- TF-IDF (1–2 grams)
- Logistic Regression

Internal HC3 test accuracy:

| Metric | Value |
|----------|----------:|
| Accuracy | 99.74% |

---

# External Benchmark Construction

## Human Texts

Source:

- MedlinePlus

Records:

| Type | Count |
|--------|--------:|
| Human | 1016 |

Characteristics:

- Official medical patient-education articles
- Written by medical organizations
- Different source from HC3

---

## AI Texts

Source:

- Generated from MedlinePlus topics

Generator:

- OpenRouter model

Records:

| Type | Count |
|--------|--------:|
| AI | 100 |

Generation protocol:

- Topic name used as prompt
- Patient-facing explanation
- 150–300 words
- Neutral medical language

---

# Human-Only Evaluation

The HC3 model was evaluated on 1016 real MedlinePlus human articles.

| Prediction | Count | Percentage |
|------------|--------:|--------:|
| Predicted Human | 398 | 39.17% |
| Predicted AI | 618 | 60.83% |

Result:

The model falsely classified 60.83% of real human articles as AI-generated.

---

# AI-Only Evaluation

The HC3 model was evaluated on 100 generated MedlinePlus AI articles.

| Prediction | Count | Percentage |
|------------|--------:|--------:|
| Predicted AI | 79 | 79.00% |
| Predicted Human | 21 | 21.00% |

Result:

The model successfully detected most generated AI articles.

---

# Combined External Benchmark

Dataset:

| Class | Count |
|---------|--------:|
| Human | 1016 |
| AI | 100 |
| Total | 1116 |

---

## Accuracy

| Metric | Value |
|----------|----------:|
| Accuracy | 42.74% |

---

## Classification Report

| Class | Precision | Recall | F1 |
|----------|----------:|----------:|----------:|
| AI | 0.11 | 0.79 | 0.20 |
| Human | 0.95 | 0.39 | 0.55 |

---

## Confusion Matrix

Rows = true labels

Columns = predicted labels

| True \\ Pred | AI | Human |
|-------------|------:|------:|
| AI | 79 | 21 |
| Human | 618 | 398 |

---

# Interpretation

The HC3-trained detector retained moderate ability to identify generated AI texts (79% recall) but exhibited extremely poor specificity on external human medical texts.

The model incorrectly labeled 618 real MedlinePlus articles as AI-generated.

This indicates that the classifier learned patterns that are not specific to AI generation and instead overlap heavily with professional medical writing.

---

# Main Finding

A model that achieved:

99.74% accuracy on HC3

collapsed to:

42.74% accuracy on an external medical benchmark.

This demonstrates that high performance on HC3 does not imply robust real-world AI-text detection.

---

# Conclusion

The external benchmark revealed severe generalization failure.

The HC3 baseline appears to rely heavily on dataset-specific stylistic artifacts and medical-writing patterns rather than robust signals of AI authorship.

Therefore, HC3 should not be used as the sole benchmark for evaluating medical AI-text detection systems.

Future experiments should focus on:

1. Cross-source training.
2. Multi-model AI generation.
3. Neural baselines (LSTM, TinyBERT).
4. Calibration and uncertainty estimation.
5. Domain-general robustness evaluation.
