import re


def word_count(text):
    return len(text.split())


def lexical_diversity(text):
    words = text.lower().split()

    if not words:
        return 0

    return len(set(words)) / len(words)


MEDICAL_WORDS = {
    "symptoms",
    "treatment",
    "condition",
    "disease",
    "medical",
    "diagnosis",
    "patient",
}


def medical_density(text):
    words = re.findall(r"\b\w+\b", text.lower())

    if not words:
        return 0

    hits = sum(1 for w in words if w in MEDICAL_WORDS)

    return hits / len(words)


def length_score(text):
    return min(word_count(text) / 500, 1.0)


def ai_confidence(
    ai_probability,
    text,
):
    return (
        0.80 * ai_probability
        + 0.10 * length_score(text)
        + 0.05 * lexical_diversity(text)
        + 0.05 * medical_density(text)
    )


def main():
    sample = """
    Diabetes is a chronic disease that affects blood sugar regulation.
    Treatment includes lifestyle changes and medication.
    """

    print("Words:", word_count(sample))
    print("TTR:", lexical_diversity(sample))
    print("Medical Density:", medical_density(sample))
    print("Length Score:", length_score(sample))

    print("AI Confidence:", ai_confidence(0.70, sample))


if __name__ == "__main__":
    main()
