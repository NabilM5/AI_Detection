"""Generate standalone Google Colab notebooks for project EDA."""

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def source(text):
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def notebook(cells, gpu=False):
    metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": []},
    }
    if gpu:
        metadata["accelerator"] = "GPU"

    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_DATA_CELLS = [
    markdown(
        """
        ## Load the project datasets

        Choose one data mode below:

        - `upload`: upload the three JSONL files from your computer.
        - `drive`: read them from Google Drive.
        - `local`: use a checked-out repository under `/content/AI_Detection`.

        Required files:

        - `hc3_medicine.jsonl`
        - `medlineplus_human.jsonl`
        - `medlineplus_ai.jsonl`
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import pandas as pd

        DATA_MODE = "upload"  # "upload", "drive", or "local"

        FILES = {
            "hc3": "hc3_medicine.jsonl",
            "medlineplus_human": "medlineplus_human.jsonl",
            "medlineplus_ai": "medlineplus_ai.jsonl",
        }

        if DATA_MODE == "upload":
            from google.colab import files

            upload_dir = Path("/content/ai_detection_data")
            upload_dir.mkdir(parents=True, exist_ok=True)
            print("Select all three JSONL files.")
            uploaded = files.upload()
            for filename, content in uploaded.items():
                (upload_dir / Path(filename).name).write_bytes(content)
            data_dir = upload_dir
        elif DATA_MODE == "drive":
            from google.colab import drive

            drive.mount("/content/drive")
            # Change this if your project is stored elsewhere in Drive.
            data_dir = Path("/content/drive/MyDrive/AI_Detection/data/processed")
        elif DATA_MODE == "local":
            data_dir = Path("/content/AI_Detection/data/processed")
        else:
            raise ValueError(f"Unsupported DATA_MODE: {DATA_MODE}")

        paths = {name: data_dir / filename for name, filename in FILES.items()}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")

        print("Data files:")
        for name, path in paths.items():
            print(f"  {name:22s} {path}")
        """
    ),
    code(
        """
        def load_jsonl(path):
            with Path(path).open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]


        def records_to_frame(records, dataset_group):
            frame = pd.DataFrame(records)
            frame["dataset_group"] = dataset_group
            return frame


        hc3 = records_to_frame(load_jsonl(paths["hc3"]), "HC3")
        mp_human = records_to_frame(
            load_jsonl(paths["medlineplus_human"]), "MedlinePlus Human"
        )
        mp_ai = records_to_frame(
            load_jsonl(paths["medlineplus_ai"]), "MedlinePlus AI"
        )

        df = pd.concat([hc3, mp_human, mp_ai], ignore_index=True, sort=False)
        df["analysis_group"] = df["dataset_group"]
        hc3_mask = df["dataset_group"].eq("HC3")
        hc3_label_names = df.loc[hc3_mask, "label"].map(
            {"human": "Human", "ai": "AI"}
        )
        if hc3_label_names.isna().any():
            unknown_index = hc3_label_names.index[hc3_label_names.isna()]
            unknown_labels = sorted(df.loc[unknown_index, "label"].unique())
            raise ValueError(f"Unknown HC3 labels: {unknown_labels}")
        df.loc[hc3_mask, "analysis_group"] = "HC3 " + hc3_label_names

        print(f"Combined rows: {len(df):,}")
        display(df.groupby(["analysis_group", "label"]).size().rename("rows").to_frame())
        display(df.head(3))
        """
    ),
]


EDA_CELLS = (
    [
        markdown(
            """
        # Medical AI Text Detection: Dataset and Linguistic EDA

        This notebook audits the HC3 Medicine and MedlinePlus datasets used by the project.
        It focuses on data quality, class/source imbalance, text length, stylometric features,
        generator provenance, statistical distribution shift, and lexical artifacts.

        **Important:** GPU is not required for this notebook. A Colab GPU runtime will still work,
        but the computations below are primarily CPU and memory bound.
        """
        ),
        code(
            """
        %pip -q install pandas seaborn matplotlib scipy scikit-learn
        """
        ),
        code(
            """
        import os
        import random
        import re
        import string
        import warnings
        from collections import Counter
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from scipy.stats import ttest_ind
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import train_test_split

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_columns", 50)

        OUTPUT_DIR = Path("/content/eda_outputs")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RANDOM_STATE = 42
        """
        ),
    ]
    + COMMON_DATA_CELLS
    + [
        markdown("## 1. Schema and data-quality audit"),
        code(
            """
        audit_rows = []
        for group, subset in df.groupby("analysis_group", dropna=False):
            audit_rows.append(
                {
                    "group": group,
                    "rows": len(subset),
                    "unique_ids": subset["id"].nunique(),
                    "duplicate_ids": subset["id"].duplicated().sum(),
                    "duplicate_texts": subset["text"].duplicated().sum(),
                    "missing_text": subset["text"].isna().sum(),
                    "empty_text": subset["text"].fillna("").str.strip().eq("").sum(),
                    "unique_source_documents": subset["source_document_id"].nunique(),
                }
            )

        audit = pd.DataFrame(audit_rows).set_index("group")
        display(audit)

        print("Columns and missing values:")
        missing_table = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(2),
            }
        ).sort_values("missing_pct", ascending=False)
        display(missing_table)

        assert df["text"].notna().all(), "Null text values found"
        assert not df["text"].str.strip().eq("").any(), "Empty text values found"
        """
        ),
        markdown("## 2. Text feature engineering"),
        code(
            """
        WORD_RE = re.compile(r"\\b\\w+\\b", flags=re.UNICODE)
        SENTENCE_RE = re.compile(r"[.!?]+")


        def text_features(text):
            text = str(text)
            words = WORD_RE.findall(text.lower())
            sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
            word_count = len(words)
            punctuation_count = sum(ch in string.punctuation for ch in text)
            unique_words = len(set(words))

            return {
                "word_count": word_count,
                "char_count": len(text),
                "sentence_count": len(sentences),
                "avg_sentence_length": word_count / max(len(sentences), 1),
                "lexical_diversity": unique_words / max(word_count, 1),
                "avg_word_length": sum(map(len, words)) / max(word_count, 1),
                "punctuation_density": punctuation_count / max(word_count, 1),
                "question_mark_count": text.count("?"),
                "starts_with_heading": int(text.lstrip().startswith("#")),
                "starts_with_greeting": int(
                    bool(re.match(r"^(hi|hello|dear|thanks|thank you)\\b", text.strip(), re.I))
                ),
            }


        feature_df = df["text"].apply(text_features).apply(pd.Series)
        eda = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

        feature_columns = list(feature_df.columns)
        summary = (
            eda.groupby("analysis_group")[feature_columns]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(3)
        )
        display(summary)
        """
        ),
        markdown("## 3. Class, source, and generator distributions"),
        code(
            """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.countplot(data=eda, x="analysis_group", hue="label", ax=axes[0])
        axes[0].set_title("Rows by source and label")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set_xlabel("")

        generator_counts = (
            eda.loc[eda["analysis_group"].eq("MedlinePlus AI"), "generator"]
            .fillna("unknown")
            .value_counts()
        )
        generator_counts.plot(kind="bar", ax=axes[1], color=sns.color_palette("viridis", len(generator_counts)))
        axes[1].set_title("MedlinePlus AI generator provenance")
        axes[1].set_ylabel("Rows")
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "class_source_generator_counts.png", dpi=160, bbox_inches="tight")
        plt.show()

        display(generator_counts.rename("rows").to_frame())
        """
        ),
        markdown("## 4. Length and stylometric distributions"),
        code(
            """
        plot_features = [
            "word_count",
            "avg_sentence_length",
            "lexical_diversity",
            "avg_word_length",
            "punctuation_density",
        ]

        fig, axes = plt.subplots(len(plot_features), 2, figsize=(16, 4 * len(plot_features)))

        for row, feature in enumerate(plot_features):
            sns.boxplot(data=eda, x="analysis_group", y=feature, ax=axes[row, 0], showfliers=False)
            axes[row, 0].tick_params(axis="x", rotation=25)
            axes[row, 0].set_xlabel("")
            axes[row, 0].set_title(f"{feature.replace('_', ' ').title()} by group")

            sns.ecdfplot(data=eda, x=feature, hue="analysis_group", ax=axes[row, 1])
            axes[row, 1].set_title(f"ECDF: {feature.replace('_', ' ')}")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "linguistic_feature_distributions.png", dpi=160, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=eda,
            x="word_count",
            hue="analysis_group",
            element="step",
            stat="density",
            common_norm=False,
            log_scale=(True, False),
        )
        plt.title("Word-count distribution (log-scaled x-axis)")
        plt.savefig(OUTPUT_DIR / "word_count_log_distribution.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
        ),
        markdown("## 5. Statistical distribution-shift tests"),
        code(
            """
        def cohens_d(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            return (a.mean() - b.mean()) / pooled if pooled else np.nan


        comparisons = [
            ("HC3 Human", "HC3 AI"),
            ("MedlinePlus Human", "MedlinePlus AI"),
            ("HC3 Human", "MedlinePlus Human"),
            ("HC3 AI", "MedlinePlus AI"),
        ]

        test_rows = []
        for left, right in comparisons:
            for feature in plot_features:
                a = eda.loc[eda["analysis_group"].eq(left), feature].dropna()
                b = eda.loc[eda["analysis_group"].eq(right), feature].dropna()
                statistic, p_value = ttest_ind(a, b, equal_var=False)
                test_rows.append(
                    {
                        "comparison": f"{left} vs {right}",
                        "feature": feature,
                        "mean_left": a.mean(),
                        "mean_right": b.mean(),
                        "welch_t": statistic,
                        "p_value": p_value,
                        "cohens_d": cohens_d(a, b),
                    }
                )

        tests = pd.DataFrame(test_rows)
        tests["significant_0.05"] = tests["p_value"] < 0.05
        display(tests.round(5))
        tests.to_csv(OUTPUT_DIR / "welch_tests_and_effect_sizes.csv", index=False)
        """
        ),
        markdown("## 6. Opening-style and lexical artifact analysis"),
        code(
            """
        def first_words(text, n=12):
            return " ".join(str(text).split()[:n]).lower()


        opening_rates = (
            eda.groupby("analysis_group")[["starts_with_greeting", "starts_with_heading"]]
            .mean()
            .mul(100)
            .round(2)
        )
        display(opening_rates.rename(columns=lambda c: c + "_pct"))

        for group in eda["analysis_group"].unique():
            print(f"\\n{group}: example openings")
            display(eda.loc[eda["analysis_group"].eq(group), "text"].head(5).map(first_words).to_frame("opening"))

        hc3_only = eda.loc[eda["dataset_group"].eq("HC3")].copy()
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=20_000, stop_words="english")
        X = vectorizer.fit_transform(hc3_only["text"])
        y = hc3_only["label"].eq("ai").astype(int)

        lexical_model = LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
        lexical_model.fit(X, y)
        names = vectorizer.get_feature_names_out()
        coefs = lexical_model.coef_[0]

        top_ai_idx = np.argsort(coefs)[-25:][::-1]
        top_human_idx = np.argsort(coefs)[:25]
        lexical_table = pd.DataFrame(
            {
                "AI-associated feature": names[top_ai_idx],
                "AI coefficient": coefs[top_ai_idx],
                "Human-associated feature": names[top_human_idx],
                "Human coefficient": coefs[top_human_idx],
            }
        )
        display(lexical_table)
        lexical_table.to_csv(OUTPUT_DIR / "hc3_lexical_artifacts.csv", index=False)
        """
        ),
        markdown("## 7. Exploratory baseline and external transfer check"),
        code(
            """
        hc3_train, hc3_test = train_test_split(
            hc3_only,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=hc3_only["label"],
        )

        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000)
        X_train = vectorizer.fit_transform(hc3_train["text"])
        X_test = vectorizer.transform(hc3_test["text"])

        model = LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
        model.fit(X_train, hc3_train["label"])

        internal_pred = model.predict(X_test)
        print("HC3 random holdout accuracy:", round(accuracy_score(hc3_test["label"], internal_pred), 4))

        external = eda.loc[eda["analysis_group"].isin(["MedlinePlus Human", "MedlinePlus AI"])].copy()
        external_pred = model.predict(vectorizer.transform(external["text"]))
        print("External MedlinePlus accuracy:", round(accuracy_score(external["label"], external_pred), 4))
        print(classification_report(external["label"], external_pred, zero_division=0))
        print("Confusion matrix labels [ai, human]:")
        print(confusion_matrix(external["label"], external_pred, labels=["ai", "human"]))

        print("\\nInterpret this as distribution-shift evidence, not a final benchmark: the random HC3 split shares source artifacts.")
        """
        ),
        markdown("## 8. Cross-source TF-IDF evaluation"),
        code(
            """
        def grouped_train_test_split(frame, train_ratio, random_state=42):
            group_ids = list(dict.fromkeys(frame["split_group"].tolist()))
            random.Random(random_state).shuffle(group_ids)
            split_index = int(len(group_ids) * train_ratio)
            train_groups = set(group_ids[:split_index])
            train = frame.loc[frame["split_group"].isin(train_groups)].copy()
            test = frame.loc[~frame["split_group"].isin(train_groups)].copy()
            return train, test


        # Reproduce the project split logic from the three uploaded base datasets.
        hc3_train, hc3_remainder = grouped_train_test_split(hc3, train_ratio=0.70)
        medlineplus_all = pd.concat([mp_human, mp_ai], ignore_index=True, sort=False)
        medlineplus_train, medlineplus_test = grouped_train_test_split(
            medlineplus_all, train_ratio=0.70
        )

        cross_train = pd.concat([hc3_train, medlineplus_train], ignore_index=True, sort=False)
        cross_test = medlineplus_test.reset_index(drop=True)

        assert set(cross_train["split_group"]).isdisjoint(set(cross_test["split_group"]))

        split_composition = (
            pd.concat(
                [
                    cross_train.assign(split="train"),
                    cross_test.assign(split="test"),
                ],
                ignore_index=True,
            )
            .groupby(["split", "source_dataset", "label"])
            .size()
            .rename("rows")
            .reset_index()
        )
        display(split_composition)
        print(f"Cross-source train rows: {len(cross_train):,}")
        print(f"Held-out MedlinePlus test rows: {len(cross_test):,}")

        cross_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            stop_words="english",
        )
        X_cross_train = cross_vectorizer.fit_transform(cross_train["text"])
        X_cross_test = cross_vectorizer.transform(cross_test["text"])

        cross_model = LogisticRegression(
            max_iter=1_000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        cross_model.fit(X_cross_train, cross_train["label"])

        cross_pred = cross_model.predict(X_cross_test)
        classes = list(cross_model.classes_)
        ai_index = classes.index("ai")
        ai_probability = cross_model.predict_proba(X_cross_test)[:, ai_index]
        y_binary = cross_test["label"].eq("ai").astype(int).to_numpy()

        cross_accuracy = accuracy_score(cross_test["label"], cross_pred)
        cross_roc_auc = roc_auc_score(y_binary, ai_probability)
        cross_confusion = confusion_matrix(
            cross_test["label"], cross_pred, labels=["ai", "human"]
        )

        print(f"Accuracy: {cross_accuracy:.4f}")
        print(f"ROC-AUC: {cross_roc_auc:.4f}")
        print(classification_report(cross_test["label"], cross_pred, zero_division=0))
        print("Confusion matrix labels [ai, human]:")
        print(cross_confusion)

        fpr, tpr, _ = roc_curve(y_binary, ai_probability)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(fpr, tpr, label=f"ROC-AUC = {cross_roc_auc:.4f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Cross-source ROC curve")
        axes[0].legend()

        sns.heatmap(
            cross_confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["AI", "Human"],
            yticklabels=["AI", "Human"],
            ax=axes[1],
        )
        axes[1].set(xlabel="Predicted", ylabel="True", title="Cross-source confusion matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cross_source_roc_confusion.png", dpi=180, bbox_inches="tight")
        plt.show()
        """
        ),
        code(
            """
        threshold_rows = []
        for threshold in np.arange(0.10, 0.61, 0.05):
            threshold_pred = np.where(ai_probability >= threshold, "ai", "human")
            precision, recall, f1, _ = precision_recall_fscore_support(
                cross_test["label"],
                threshold_pred,
                labels=["ai", "human"],
                zero_division=0,
            )
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy_score(cross_test["label"], threshold_pred),
                    "ai_precision": precision[0],
                    "ai_recall": recall[0],
                    "ai_f1": f1[0],
                    "human_recall": recall[1],
                }
            )

        threshold_table = pd.DataFrame(threshold_rows)
        display(threshold_table.round(4))
        threshold_table.to_csv(OUTPUT_DIR / "cross_source_threshold_sweep.csv", index=False)

        feature_names = cross_vectorizer.get_feature_names_out()
        coefficients = cross_model.coef_[0]
        # sklearn classes are ['ai', 'human']; positive coefficients indicate human.
        top_human = np.argsort(coefficients)[-25:][::-1]
        top_ai = np.argsort(coefficients)[:25]
        cross_features = pd.DataFrame(
            {
                "human_feature": feature_names[top_human],
                "human_coefficient": coefficients[top_human],
                "ai_feature": feature_names[top_ai],
                "ai_coefficient": coefficients[top_ai],
            }
        )
        display(cross_features)
        cross_features.to_csv(OUTPUT_DIR / "cross_source_top_features.csv", index=False)

        cross_metrics = pd.DataFrame(
            [
                {
                    "train_rows": len(cross_train),
                    "test_rows": len(cross_test),
                    "accuracy": cross_accuracy,
                    "roc_auc": cross_roc_auc,
                    "ai_true_positive": int(cross_confusion[0, 0]),
                    "ai_false_negative": int(cross_confusion[0, 1]),
                    "human_false_positive": int(cross_confusion[1, 0]),
                    "human_true_negative": int(cross_confusion[1, 1]),
                }
            ]
        )
        display(cross_metrics.round(4))
        cross_metrics.to_csv(OUTPUT_DIR / "cross_source_metrics.csv", index=False)
        """
        ),
        markdown(
            """
        ### Cross-source interpretation

        The expected current-project result is approximately **94.66% accuracy** and
        **95.87% ROC-AUC** on 337 held-out MedlinePlus rows. At the default threshold,
        human recall is very high but AI recall is low. The threshold sweep shows why
        calibration is necessary, while the coefficient table reveals that source/style
        artifacts remain present even after cross-source training.
        """
        ),
        markdown("## 9. Export EDA tables and figures"),
        code(
            """
        export_columns = [
            "id", "label", "dataset_group", "analysis_group", "source_document_id",
            "medical_topic", "generator", *feature_columns,
        ]
        eda[[c for c in export_columns if c in eda.columns]].to_csv(
            OUTPUT_DIR / "text_level_eda_features.csv", index=False
        )
        audit.to_csv(OUTPUT_DIR / "dataset_audit.csv")
        summary.to_csv(OUTPUT_DIR / "group_feature_summary.csv")

        import shutil
        archive = shutil.make_archive("/content/medical_text_eda_outputs", "zip", OUTPUT_DIR)
        print("Created:", archive)
        print("Output files:")
        for path in sorted(OUTPUT_DIR.iterdir()):
            print(" -", path.name)

        """
        ),
        markdown(
            """
        ## Interpretation checklist

        Before drawing conclusions, check:

        1. Are source and label confounded?
        2. Are human and AI texts length-matched?
        3. Are paired texts kept in the same split via `source_document_id`?
        4. Does performance transfer from HC3 to MedlinePlus?
        5. Are generator-specific artifacts driving the result?
        6. Is class imbalance making accuracy misleading?
        """
        ),
    ]
)


EMBEDDING_CELLS = (
    [
        markdown(
            """
        # Medical AI Text Detection: GPU Embedding EDA

        This standalone Colab notebook uses a sentence-transformer on GPU to inspect the
        semantic geometry of HC3 Medicine and MedlinePlus text. It visualizes embeddings,
        measures group separation, finds nearest neighbors, and tests whether an embedding
        classifier trained on HC3 transfers to MedlinePlus.

        In Colab select **Runtime → Change runtime type → T4 GPU** before running all cells.
        """
        ),
        code(
            """
        %pip -q install sentence-transformers umap-learn seaborn scikit-learn
        """
        ),
        code(
            """
        import os
        import warnings
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import umap
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import NearestNeighbors

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", context="notebook")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("PyTorch:", torch.__version__)
        print("Device:", DEVICE)
        if DEVICE == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("Warning: no GPU detected; embeddings will run on CPU.")

        OUTPUT_DIR = Path("/content/embedding_eda_outputs")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RANDOM_STATE = 42
        """
        ),
    ]
    + COMMON_DATA_CELLS
    + [
        markdown("## 1. Build a balanced exploratory sample"),
        code(
            """
        MAX_PER_GROUP = 500

        sampled_parts = []
        for group, subset in df.groupby("analysis_group"):
            n = min(len(subset), MAX_PER_GROUP)
            sampled_parts.append(subset.sample(n=n, random_state=RANDOM_STATE))

        sample = pd.concat(sampled_parts, ignore_index=True)
        sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        display(sample["analysis_group"].value_counts().rename("rows").to_frame())
        print("Embedding rows:", len(sample))
        """
        ),
        markdown("## 2. Encode texts on GPU"),
        code(
            """
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        BATCH_SIZE = 64 if DEVICE == "cuda" else 16

        encoder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        embeddings = encoder.encode(
            sample["text"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        print("Embedding shape:", embeddings.shape)
        print("Embedding dtype:", embeddings.dtype)
        """
        ),
        markdown("## 3. PCA and UMAP projections"),
        code(
            """
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_xy = pca.fit_transform(embeddings)

        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.15,
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        umap_xy = reducer.fit_transform(embeddings)

        projection = sample[["id", "label", "analysis_group", "source_document_id", "generator"]].copy()
        projection[["pca_1", "pca_2"]] = pca_xy
        projection[["umap_1", "umap_2"]] = umap_xy

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.scatterplot(data=projection, x="pca_1", y="pca_2", hue="analysis_group", alpha=0.65, s=35, ax=axes[0])
        axes[0].set_title(f"PCA (explained variance: {pca.explained_variance_ratio_.sum():.1%})")

        sns.scatterplot(data=projection, x="umap_1", y="umap_2", hue="analysis_group", alpha=0.65, s=35, ax=axes[1])
        axes[1].set_title("UMAP of normalized sentence embeddings")

        for ax in axes:
            ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "embedding_pca_umap.png", dpi=180, bbox_inches="tight")
        plt.show()
        """
        ),
        markdown("## 4. Group-centroid cosine similarity"),
        code(
            """
        groups = sorted(sample["analysis_group"].unique())
        centroids = []
        for group in groups:
            centroid = embeddings[sample["analysis_group"].eq(group).to_numpy()].mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            centroids.append(centroid)

        centroid_similarity = pd.DataFrame(
            cosine_similarity(np.vstack(centroids)),
            index=groups,
            columns=groups,
        )
        display(centroid_similarity.round(3))

        plt.figure(figsize=(8, 6))
        sns.heatmap(centroid_similarity, annot=True, fmt=".3f", cmap="mako", vmin=0, vmax=1)
        plt.title("Cosine similarity between group centroids")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "centroid_cosine_similarity.png", dpi=180, bbox_inches="tight")
        plt.show()
        """
        ),
        markdown("## 5. Nearest-neighbor inspection"),
        code(
            """
        neighbors = NearestNeighbors(n_neighbors=6, metric="cosine").fit(embeddings)
        distances, indices = neighbors.kneighbors(embeddings)

        neighbor_rows = []
        for row_index in range(len(sample)):
            for rank, (neighbor_index, distance) in enumerate(zip(indices[row_index, 1:], distances[row_index, 1:]), start=1):
                neighbor_rows.append(
                    {
                        "query_id": sample.iloc[row_index]["id"],
                        "query_group": sample.iloc[row_index]["analysis_group"],
                        "neighbor_id": sample.iloc[neighbor_index]["id"],
                        "neighbor_group": sample.iloc[neighbor_index]["analysis_group"],
                        "rank": rank,
                        "cosine_similarity": 1 - distance,
                    }
                )

        neighbor_df = pd.DataFrame(neighbor_rows)
        cross_group = neighbor_df.loc[neighbor_df["query_group"] != neighbor_df["neighbor_group"]]
        display(cross_group.sort_values("cosine_similarity", ascending=False).head(30))

        same_group_rate = (neighbor_df["query_group"] == neighbor_df["neighbor_group"]).mean()
        print(f"Nearest-neighbor same-group rate across top 5: {same_group_rate:.2%}")
        neighbor_df.to_csv(OUTPUT_DIR / "nearest_neighbors.csv", index=False)
        """
        ),
        markdown("## 6. Embedding classifier: internal vs external transfer"),
        code(
            """
        hc3_mask = sample["analysis_group"].isin(["HC3 Human", "HC3 AI"]).to_numpy()
        mp_mask = sample["analysis_group"].isin(["MedlinePlus Human", "MedlinePlus AI"]).to_numpy()

        hc3_indices = np.flatnonzero(hc3_mask)
        hc3_class_counts = sample.iloc[hc3_indices]["label"].value_counts()
        print("HC3 rows selected for embedding classifier:")
        display(hc3_class_counts.rename("rows").to_frame())
        if len(hc3_class_counts) < 2:
            raise ValueError(
                "HC3 embedding subset must contain both human and AI labels. "
                f"Found: {hc3_class_counts.to_dict()}"
            )

        train_idx, test_idx = train_test_split(
            hc3_indices,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=sample.iloc[hc3_indices]["label"],
        )

        classifier = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)
        classifier.fit(embeddings[train_idx], sample.iloc[train_idx]["label"])

        internal_pred = classifier.predict(embeddings[test_idx])
        print("HC3 embedding holdout accuracy:", round(accuracy_score(sample.iloc[test_idx]["label"], internal_pred), 4))
        print(classification_report(sample.iloc[test_idx]["label"], internal_pred, zero_division=0))

        mp_idx = np.flatnonzero(mp_mask)
        external_pred = classifier.predict(embeddings[mp_idx])
        external_true = sample.iloc[mp_idx]["label"]

        print("MedlinePlus transfer accuracy:", round(accuracy_score(external_true, external_pred), 4))
        print(classification_report(external_true, external_pred, zero_division=0))
        print("Confusion matrix labels [ai, human]:")
        print(confusion_matrix(external_true, external_pred, labels=["ai", "human"]))
        """
        ),
        markdown("## 7. Source separability diagnostic"),
        code(
            """
        source_labels = sample["analysis_group"].to_numpy()
        train_idx, test_idx = train_test_split(
            np.arange(len(sample)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=source_labels,
        )

        source_classifier = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)
        source_classifier.fit(embeddings[train_idx], source_labels[train_idx])
        source_pred = source_classifier.predict(embeddings[test_idx])

        print("Source/group prediction accuracy:", round(accuracy_score(source_labels[test_idx], source_pred), 4))
        print(classification_report(source_labels[test_idx], source_pred, zero_division=0))
        print("\\nHigh source accuracy indicates strong domain structure that an AI detector can exploit accidentally.")
        """
        ),
        markdown("## 8. Export embeddings and projections"),
        code(
            """
        projection.to_csv(OUTPUT_DIR / "embedding_projection.csv", index=False)
        np.savez_compressed(
            OUTPUT_DIR / "sentence_embeddings.npz",
            embeddings=embeddings,
            ids=sample["id"].to_numpy(),
            groups=sample["analysis_group"].to_numpy(),
            labels=sample["label"].to_numpy(),
        )

        import shutil
        archive = shutil.make_archive("/content/medical_embedding_eda_outputs", "zip", OUTPUT_DIR)
        print("Created:", archive)
        for path in sorted(OUTPUT_DIR.iterdir()):
            print(" -", path.name)

        """
        ),
        markdown(
            """
        ## Interpretation

        Clear clusters do not prove that embeddings encode AI authorship. They may encode source,
        genre, length, formatting, or generator family. Compare internal HC3 accuracy with external
        MedlinePlus transfer and source-classification accuracy before claiming generalization.
        """
        ),
    ]
)


def write_notebook(filename, cells, gpu=False):
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(notebook(cells, gpu=gpu), f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"Created {path.relative_to(ROOT)}")


def main():
    write_notebook("01_dataset_linguistic_eda_colab.ipynb", EDA_CELLS)
    write_notebook("02_gpu_embedding_eda_colab.ipynb", EMBEDDING_CELLS, gpu=True)


if __name__ == "__main__":
    main()
