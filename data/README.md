# Data

Datasets are generated locally and intentionally excluded from Git.

Expected layout after running the preparation scripts:

```text
data/
├── external/medlineplus/mplus_topics.xml
├── raw/hc3_medicine_raw.json
└── processed/
    ├── hc3_medicine.jsonl
    ├── medlineplus_human.jsonl
    ├── medlineplus_ai.jsonl
    ├── splits/
    ├── medlineplus_splits/
    └── cross_source/
```

## Sources

- **HC3 Medicine**: human and ChatGPT medical question-answer pairs from the HC3 corpus.
- **MedlinePlus**: English patient-education topics distributed by the U.S. National Library of Medicine.
- **Generated MedlinePlus texts**: LLM-written patient-facing explanations generated from topic names. Each JSONL record stores its generator and provenance metadata.

## Build

Run commands from the repository root:

```bash
python scripts/download_med_data.py
python src/data/prepare_hc3_jsonl.py
python src/data/make_splits.py
python src/data/download_medlineplus.py
python src/data/generate_medlineplus_ai.py
python src/data/make_medlineplus_splits.py
python src/data/make_cross_source_splits.py
```

Review the upstream dataset terms and MedlinePlus attribution requirements before redistributing any data.
