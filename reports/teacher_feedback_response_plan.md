# Response to Supervisor Feedback

## User Scenario

The project focuses on real-time detection of AI-generated patient-facing medical answers.

A realistic scenario is a medical Q&A or patient-education platform where answers may be written by doctors, editors, AI assistants, or a mixture of human and AI writing. Before publication, the system estimates whether the text is human-written, AI-generated, or uncertain, and provides a confidence score for moderation.

The goal is not automatic punishment or deletion, but decision support for human reviewers.

## Dataset Strategy

The initial dataset is HC3 Medicine because it contains paired human and ChatGPT answers for the same medical questions.

However, initial experiments showed that HC3 contains strong stylistic artifacts. Therefore, HC3 is used only as a first controlled baseline.

To test generalization, an external benchmark was built using:

- MedlinePlus human patient-education articles
- AI-generated articles from the same MedlinePlus topics

Future extensions may include:

- ClinicalTrials.gov summaries
- PMC abstracts with reuse-compatible licenses
- Clinical notes only if controlled access is obtained

## Current Findings

TF-IDF + Logistic Regression achieved 99.74% accuracy on HC3 Medicine.

However, artifact analysis showed that the model relied heavily on opening-style features such as:

- hi
- hello
- thanks
- query
- regards
- dr

For AI answers, the model relied on phrases such as:

- possible
- important
- potential causes
- experiencing
- appropriate treatment

A length-only model achieved 84.97%, showing that text length is also a strong artifact.

A truncation study showed that the first 10 words already give 94.56% accuracy, confirming that HC3 contains strong opening-style artifacts.

External validation on MedlinePlus showed poor generalization. The HC3-trained model classified 60.83% of real MedlinePlus human articles as AI-generated.

## Why TF-IDF Is Still Useful

TF-IDF is not used as the final model.

It is used as an interpretable diagnostic baseline to expose dataset artifacts before training neural models.

This helps prevent misleading conclusions from high benchmark accuracy.

## Next Models

Following the supervisor's recommendation, the next stage will implement:

1. fastText-style embeddings + LSTM
2. GRU comparison
3. TinyBERT baseline

The LSTM/GRU models will test whether local context improves robustness compared with TF-IDF.

TinyBERT will test whether a compact transformer can improve generalization.

## Evaluation Metrics

The project will report:

- Accuracy
- Precision
- Recall
- F1-score
- False positive rate on human medical texts
- AI recall
- Calibration / confidence quality

False positive rate is especially important because incorrectly labeling human medical content as AI-generated is harmful in a real moderation scenario.

## Soft Label / Confidence Plan

The final system will not only output a hard label.

It will output an AI-confidence score based on:

- model probability
- calibration
- uncertainty
- artifact indicators
- optional statistical/topological features

This can later support mixed-content detection.
