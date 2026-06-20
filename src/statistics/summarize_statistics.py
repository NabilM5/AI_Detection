import pandas as pd


CSV_PATH = "reports/statistics/text_statistics.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    groups = [
        ("hc3_medicine", "human"),
        ("hc3_medicine", "ai"),
        ("medlineplus_human", "human"),
        ("medlineplus_ai", "ai"),
    ]

    print("\nSTATISTICAL SUMMARY")
    print("=" * 80)

    for dataset, label in groups:
        subset = df[(df["dataset"] == dataset) & (df["label"] == label)]

        print(f"\nDataset: {dataset}")
        print(f"Label: {label}")
        print("-" * 50)

        print(f"Documents: {len(subset)}")

        print(f"Word count: " f"{subset['word_count'].mean():.2f}")

        print(f"Sentence count: " f"{subset['sentence_count'].mean():.2f}")

        print(f"Avg sentence length: " f"{subset['avg_sentence_length'].mean():.2f}")

        print(f"Lexical diversity: " f"{subset['lexical_diversity'].mean():.4f}")

        print(f"Avg word length: " f"{subset['avg_word_length'].mean():.2f}")

        print(f"Punctuation density: " f"{subset['punctuation_density'].mean():.4f}")


if __name__ == "__main__":
    main()
