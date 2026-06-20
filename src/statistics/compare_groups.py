import pandas as pd
from scipy.stats import ttest_ind


CSV_PATH = "reports/statistics/text_statistics.csv"


def compare(df, dataset1, label1, dataset2, label2, feature):
    group1 = df[(df["dataset"] == dataset1) & (df["label"] == label1)][feature]

    group2 = df[(df["dataset"] == dataset2) & (df["label"] == label2)][feature]

    stat, p_value = ttest_ind(group1, group2, equal_var=False)

    print(f"\n{dataset1}/{label1} vs {dataset2}/{label2}")
    print(f"Feature: {feature}")
    print(f"Mean 1: {group1.mean():.4f}")
    print(f"Mean 2: {group2.mean():.4f}")
    print(f"T-statistic: {stat:.4f}")
    print(f"P-value: {p_value:.10f}")


def main():
    df = pd.read_csv(CSV_PATH)

    features = [
        "word_count",
        "avg_sentence_length",
        "lexical_diversity",
    ]

    comparisons = [
        ("hc3_medicine", "human", "hc3_medicine", "ai"),
        ("medlineplus_human", "human", "medlineplus_ai", "ai"),
        ("hc3_medicine", "human", "medlineplus_human", "human"),
        ("hc3_medicine", "ai", "medlineplus_ai", "ai"),
    ]

    print("\nSTATISTICAL COMPARISON")
    print("=" * 80)

    for comparison in comparisons:
        d1, l1, d2, l2 = comparison

        print("\n" + "=" * 80)

        for feature in features:
            compare(
                df,
                d1,
                l1,
                d2,
                l2,
                feature,
            )


if __name__ == "__main__":
    main()
