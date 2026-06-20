# Experiment 01 — HC3 Medicine Baseline

## Goal

Evaluate whether a simple classical machine learning model can distinguish human medical answers from ChatGPT-generated medical answers in the HC3 medicine subset.

## Dataset

Source: Hello-SimpleAI/HC3, medicine subset.

After preprocessing:

| Class | Count |
|---|---:|
| Human | 1,248 |
| AI | 1,334 |
| Total | 2,582 |

Grouped split by question ID:

| Split | Records |
|---|---:|
| Train | 1,806 |
| Validation | 390 |
| Test | 386 |

The split was grouped by `split_group` so that human and AI answers for the same medical question stayed in the same split.

## EDA

| Class | Avg Words | Min | Max |
|---|---:|---:|---:|
| Human | 82.22 | 1 | 363 |
| AI | 186.82 | 10 | 407 |

AI answers are more than twice as long as human answers on average, which indicates a potential length-based artifact.

## Baseline Results

| Model | Accuracy |
|---|---:|
| Length-only Logistic Regression | 84.97% |
| TF-IDF + Logistic Regression | 99.74% |
| TF-IDF + Logistic Regression, first 100 words | 98.19% |

## Truncation Sweep

| Words Kept | Accuracy |
|---:|---:|
| 10 | 94.56% |
| 25 | 94.56% |
| 50 | 96.63% |
| 100 | 98.19% |
| 200 | 98.70% |
| Full text | 99.74% |

## Feature Analysis

The 10-word TF-IDF model achieved 94.56% accuracy. Feature inspection showed that the classifier mainly used opening-style artifacts.

Human-associated opening features:

- hi
- hello
- thanks
- query
- welcome
- dear
- hcm
- dr

AI-associated opening features:

- possible
- sorry
- important
- sorry hear
- generally
- potential causes
- experiencing
- medical
- appropriate
- treatment

## External Validation on MedlinePlus

A model trained on HC3 Medicine was evaluated on 1,016 real human MedlinePlus patient-education articles.

| Prediction | Count | Percentage |
|---|---:|---:|
| Predicted Human | 398 | 39.17% |
| Predicted AI | 618 | 60.83% |

The model falsely classified 60.83% of MedlinePlus human articles as AI-generated.

## MedlinePlus Length Analysis

| Length Bucket | Predicted AI |
|---|---:|
| 0–99 words | 42.65% |
| 100–199 words | 52.47% |
| 200–399 words | 59.73% |
| 400–699 words | 75.86% |
| 700+ words | 77.50% |

Longer MedlinePlus human articles were more likely to be classified as AI.

## Conclusion

Although TF-IDF + Logistic Regression achieved 99.74% accuracy on the HC3 Medicine test set, deeper analysis showed that the result is strongly influenced by dataset artifacts.

The length-only model achieved 84.97%, showing that answer length contains substantial label information. However, truncating texts to the first 100 words still produced 98.19% accuracy, and even the first 10 words gave 94.56% accuracy. This suggests that opening style and lexical artifacts are even stronger than length alone.

External validation on MedlinePlus confirmed poor generalization: 60.83% of real human MedlinePlus articles were incorrectly classified as AI-generated.

Therefore, high HC3 accuracy should not be interpreted as evidence of robust AI-generated medical text detection. The baseline primarily learns HC3-specific stylistic artifacts, especially forum-doctor greetings versus assistant-style openings.
