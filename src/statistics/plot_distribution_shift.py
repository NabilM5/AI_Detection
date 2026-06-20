import os

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = "reports/statistics/text_statistics.csv"
OUTPUT_DIR = "reports/statistics/figures"


def plot_feature(df, feature, filename):
    plt.figure(figsize=(10, 6))

    groups = [
        ("hc3_medicine", "human", "HC3 Human"),
        ("hc3_medicine", "ai", "HC3 AI"),
        ("medlineplus_human", "human", "MedlinePlus Human"),
        ("medlineplus_ai", "ai", "MedlinePlus AI"),
    ]

    data = []

    for dataset, label, title in groups:
        subset = df[(df["dataset"] == dataset) & (df["label"] == label)]

        data.append(subset[feature])

    plt.boxplot(
        data,
        tick_labels=[
            "HC3 Human",
            "HC3 AI",
            "MP Human",
            "MP AI",
        ],
    )

    plt.title(feature.replace("_", " ").title())
    plt.ylabel(feature)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    plot_feature(
        df,
        "word_count",
        "word_count_distribution.png",
    )

    plot_feature(
        df,
        "lexical_diversity",
        "lexical_diversity_distribution.png",
    )

    plot_feature(
        df,
        "avg_sentence_length",
        "sentence_length_distribution.png",
    )


if __name__ == "__main__":
    main()
