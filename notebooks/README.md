# Colab Notebooks

## 1. Dataset and Linguistic EDA

[`01_dataset_linguistic_eda_colab.ipynb`](01_dataset_linguistic_eda_colab.ipynb)

Audits schemas, missing values, duplicate IDs/texts, source and label balance, generator provenance, text lengths, stylometric features, Welch tests, effect sizes, and HC3 lexical artifacts. It also runs an exploratory HC3-to-MedlinePlus transfer check and reproduces the cross-source TF-IDF experiment, including ROC-AUC, confusion matrix, threshold sweep, and coefficient analysis. All figures and tables are exported as a ZIP file.

GPU is optional for this notebook.

## 2. GPU Embedding EDA

[`02_gpu_embedding_eda_colab.ipynb`](02_gpu_embedding_eda_colab.ipynb)

Uses `sentence-transformers/all-MiniLM-L6-v2` on a Colab GPU to produce normalized text embeddings, PCA/UMAP plots, group-centroid similarities, nearest-neighbor diagnostics, an HC3-to-MedlinePlus transfer experiment, and a source-separability diagnostic.

Select **Runtime → Change runtime type → T4 GPU** before running it.

## Data Input

Both notebooks are standalone and default to `DATA_MODE = "upload"`. Upload:

- `data/processed/hc3_medicine.jsonl`
- `data/processed/medlineplus_human.jsonl`
- `data/processed/medlineplus_ai.jsonl`

They also support Google Drive and a repository checked out under `/content/AI_Detection`.

## Regeneration

The notebooks are generated from a version-controlled script:

```bash
python scripts/create_eda_notebooks.py
```
