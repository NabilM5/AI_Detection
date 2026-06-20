const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat, PageBreak
} = require('docx');
const fs = require('fs');

// Document theme
const TEAL       = "007A8A";
const DARK_NAVY  = "1A2A3A";
const MID_GREY   = "555555";
const LIGHT_GREY = "EEEEEE";
const PALE_TEAL  = "E6F4F6";
const WHITE      = "FFFFFF";

// A4 layout
const PAGE_W    = 11906;
const PAGE_H    = 16838;
const MARGIN    = 1134;
const CONTENT_W = PAGE_W - MARGIN * 2;

const noBorder  = { style: BorderStyle.NONE, size: 0, color: WHITE };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const thinBorder = { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" };
const thinBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };

function spacer(pt = 6) {
  return new Paragraph({ spacing: { before: 0, after: 0, line: pt * 20 }, children: [new TextRun("")] });
}

function hrRule(colorHex = TEAL) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: colorHex, space: 1 } },
    children: [new TextRun("")]
  });
}

function sectionHeading(text) {
  return new Paragraph({
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text: text.toUpperCase(), bold: true, size: 28, color: TEAL, font: "Arial" })]
  });
}

function subsectionHeading(text) {
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, bold: true, size: 24, color: DARK_NAVY, font: "Arial" })]
  });
}

function bodyText(text) {
  return new Paragraph({
    spacing: { before: 40, after: 60 },
    children: [new TextRun({ text, size: 22, color: DARK_NAVY, font: "Arial" })]
  });
}

function bullet(text) {
  return new Paragraph({
    spacing: { before: 30, after: 30 },
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, size: 22, color: DARK_NAVY, font: "Arial" })]
  });
}

function infoBox(label, text) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [1800, CONTENT_W - 1800],
    borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideH: noBorder, insideV: noBorder },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: noBorders,
            width: { size: 1800, type: WidthType.DXA },
            shading: { fill: "007A8A", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 120, right: 80 },
            children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 20, color: WHITE, font: "Arial" })] })]
          }),
          new TableCell({
            borders: noBorders,
            width: { size: CONTENT_W - 1800, type: WidthType.DXA },
            shading: { fill: PALE_TEAL, type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: [new Paragraph({ children: [new TextRun({ text, size: 20, color: DARK_NAVY, font: "Arial" })] })]
          })
        ]
      })
    ]
  });
}

function phaseBox(phaseNum, title, duration, tasks) {
  const headerCell = new TableCell({
    borders: noBorders,
    width: { size: CONTENT_W, type: WidthType.DXA },
    shading: { fill: "007A8A", type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 160, right: 160 },
    children: [
      new Paragraph({
        children: [
          new TextRun({ text: `Phase ${phaseNum}: ${title}`, bold: true, size: 24, color: WHITE, font: "Arial" }),
          new TextRun({ text: `   ·   ${duration}`, size: 20, color: "CCEEEE", font: "Arial" }),
        ]
      })
    ]
  });

  const bodyRows = tasks.map(t => new TableRow({
    children: [
      new TableCell({
        borders: noBorders,
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { fill: PALE_TEAL, type: ShadingType.CLEAR },
        margins: { top: 40, bottom: 40, left: 160, right: 160 },
        children: [new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: t, size: 21, color: DARK_NAVY, font: "Arial" })]
        })]
      })
    ]
  }));

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: TEAL },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: TEAL },
      left: { style: BorderStyle.SINGLE, size: 4, color: TEAL },
      right: { style: BorderStyle.SINGLE, size: 4, color: TEAL },
      insideH: noBorder, insideV: noBorder
    },
    rows: [new TableRow({ children: [headerCell] }), ...bodyRows]
  });
}

function metricsTable() {
  const colWidths = [1800, 3800, CONTENT_W - 5600];
  const headers = ["Metric", "Description", "Why It Matters"];
  const rows = [
    ["Accuracy", "Fraction of correct predictions", "Primary performance indicator"],
    ["F1-Score", "Harmonic mean of precision and recall", "Handles class imbalance"],
    ["ROC-AUC", "Area under the ROC curve", "Threshold-independent quality measure"],
    ["Confidence Score", "Model certainty per sample (0–1)", "Soft labels — key research goal"],
    ["Topological Complexity", "Number of holes (Betti numbers) in data", "Distinguishes human vs AI structure"],
    ["Hausdorff Dimension", "Fractal dimension of text", "Self-organised criticality criterion (Gromov)"],
  ];

  const header = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders: thinBorders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: "007A8A", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 20, color: WHITE, font: "Arial" })] })]
    }))
  });

  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, i) => new TableCell({
      borders: thinBorders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: i === 0 ? PALE_TEAL : WHITE, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, color: DARK_NAVY, font: "Arial", bold: i === 0 })] })]
    }))
  }));

  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...dataRows] });
}

function modelsTable() {
  const colWidths = [2200, 2600, 1800, CONTENT_W - 6600];
  const headers = ["Model", "Type", "Complexity", "Rationale"];
  const rows = [
    ["TF-IDF + SVM", "Classical ML (baseline)", "Low", "Standard lower bound; logically won't yield high results (supervisor's note) — included to confirm"],
    ["fastText + LSTM", "Hybrid (embeddings + sequence)", "Medium", "Recommended by supervisor as strong baseline — captures local context and morphology"],
    ["GRU vs LSTM", "RNN architecture comparison", "Medium", "Experiment: which recurrent cell better captures AI text patterns?"],
    ["TinyBERT", "Transformer (distilled)", "High", "Contextual embeddings; expected F1 leader"],
    ["TDA Pipeline", "Topological analysis", "Experimental", "Perforation metric + Betti numbers (Hidden Holes paper) + Hausdorff dimension (Gromov)"],
  ];

  const header = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders: thinBorders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: "007A8A", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 20, color: WHITE, font: "Arial" })] })]
    }))
  });

  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, i) => new TableCell({
      borders: thinBorders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: ri % 2 === 0 ? WHITE : "F5FAFB", type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, color: DARK_NAVY, font: "Arial", bold: i === 0 })] })]
    }))
  }));

  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...dataRows] });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 480, hanging: 280 } } } }]
      },
      {
        reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 480, hanging: 280 } } } }]
      },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22, color: DARK_NAVY } } }
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN }
      }
    },
    children: [

      // Title
      new Paragraph({
        spacing: { before: 0, after: 20 },
        children: [new TextRun({ text: "RESEARCH PROJECT PLAN", size: 20, color: MID_GREY, font: "Arial", bold: true })]
      }),
      new Paragraph({
        spacing: { before: 0, after: 40 },
        children: [new TextRun({ text: "Seminar Project  ·  HSE University  ·  2026", size: 20, color: MID_GREY, font: "Arial" })]
      }),
      hrRule(TEAL),
      new Paragraph({
        spacing: { before: 80, after: 60 },
        children: [new TextRun({ text: "Detection of", bold: true, size: 52, color: DARK_NAVY, font: "Arial" })]
      }),
      new Paragraph({
        spacing: { before: 0, after: 60 },
        children: [new TextRun({ text: "AI-Generated Text", bold: true, size: 52, color: DARK_NAVY, font: "Arial" })]
      }),
      new Paragraph({
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Classical ML  ·  Neural Networks  ·  Topological Data Analysis", size: 26, color: TEAL, font: "Arial" })]
      }),
      hrRule(LIGHT_GREY),
      spacer(6),
      infoBox("Author", "Nabil Mouhamech  ·  Master's in Applied AI"),
      spacer(4),
      infoBox("University", "HSE — Higher School of Economics  ·  Moscow"),
      spacer(4),
      infoBox("Type", "Seminar Research Project"),
      spacer(4),
      infoBox("Status", "Planning / In Progress"),
      spacer(4),
      infoBox("Year", "2026"),
      spacer(12),

      // 1. Rationale
      sectionHeading("1. Rationale and Motivation"),
      hrRule(TEAL),

      bodyText("The widespread adoption of large language models — GPT-4, LLaMA, Gemini — creates a growing threat of misuse: academic plagiarism, disinformation, and fabricated content. Automated detection of AI-generated text is becoming a critical research challenge."),
      spacer(4),
      bodyText("This project compares classical ML approaches with neural network methods for binary classification of human vs AI-generated text. It also explores Topological Data Analysis (TDA) as an alternative approach grounded in the structural properties of language — capable of 93–97% accuracy without any neural network."),
      spacer(4),
      subsectionHeading("Core Research Questions"),
      bullet("Which approach performs best: classical ML (TF-IDF + SVM) or neural (fastText+LSTM, TinyBERT)?"),
      bullet("How does the topological complexity of AI-generated text differ from human-written text?"),
      bullet("Is the 'perforation metric' from the Hidden Holes paper applicable to AI detection?"),
      bullet("Which RNN architecture better captures AI text patterns — GRU or LSTM?"),
      bullet("Can we generate soft labels — a confidence score rather than a hard binary decision?"),
      spacer(8),

      // 2. Theoretical foundations
      sectionHeading("2. Theoretical Foundations"),
      hrRule(TEAL),

      subsectionHeading("2.1  \"Hidden Holes\" — Topological Analysis of Text"),
      bodyText("The paper (Kushnareva et al.) demonstrates that human-written texts exhibit higher topological complexity than AI-generated ones. This is measured using persistent homology and Vietoris-Rips complexes built on high-dimensional text embeddings."),
      spacer(4),
      bullet("Perforation metric: measures the number and persistence of topological holes (Betti numbers β₁) in the embedding space"),
      bullet("Key finding: LSTM-generated text >> Transformer-generated text in topological complexity, explaining why LSTM outputs are harder to distinguish from human text"),
      bullet("Practical application: compute the perforation metric as an additional feature for the classifier"),
      spacer(8),

      subsectionHeading("2.2  \"Spot the Bot\" — Gromov's Method (HSE)"),
      bodyText("Gromov treats language as a self-organised critical system — analogous to physical systems at a phase transition boundary. AI-generated text violates this criticality, which manifests as measurable statistical anomalies."),
      spacer(4),
      bullet("Hausdorff dimension: fractal characteristic of text; AI texts systematically differ from human texts on this measure"),
      bullet("Entropy-complexity plane: AI texts cluster in a specific region of this 2D space"),
      bullet("TDA without neural networks: method achieves 93–97% accuracy using only topological features"),
      bullet("Diagnostic potential: detects stylometric deviations and bot authorship"),
      spacer(8),

      subsectionHeading("2.3  Supervisor's Recommendations"),
      bodyText("The HSE supervisor provided specific guidance on developing this project:"),
      spacer(4),
      bullet("User scenarios: describe concrete use cases — student essays, news articles, product reviews"),
      bullet("Dataset quality: pay close attention to training data quality — garbage in, garbage out"),
      bullet("Statistical and topological methods: apply TDA alongside neural approaches"),
      bullet("Baseline: fastText + LSTM — recommended as a strong baseline that captures local context"),
      bullet("Architecture comparison: GRU vs LSTM — run a controlled experiment"),
      bullet("Advanced model: try TinyBERT — a distilled BERT variant"),
      bullet("Soft labels: design a confidence metric instead of hard binary classification"),
      bullet("TF-IDF: logically cannot yield high results — include as lower bound to confirm"),
      spacer(8),

      // 3. User scenarios
      sectionHeading("3. User Scenarios"),
      hrRule(TEAL),

      bodyText("As directed by the supervisor, the specific use cases and user profiles must be clearly defined."),
      spacer(6),

      subsectionHeading("Scenario A: Academic Settings"),
      bullet("User: lecturer, teaching assistant, anti-plagiarism platform"),
      bullet("Input: student essay, coursework, or seminar paper"),
      bullet("Output: AI-generation probability + highlighted high-confidence passages"),
      bullet("Requirement: high recall (false alarm preferable to missing AI text)"),
      spacer(6),

      subsectionHeading("Scenario B: Media & Fact-Checking"),
      bullet("User: editor, fact-checker, publishing platform"),
      bullet("Input: news article, press release, social media post"),
      bullet("Output: credibility score + source classification (human / AI / mixed)"),
      bullet("Requirement: high precision, explainability of the decision"),
      spacer(6),

      subsectionHeading("Scenario C: Review Platforms"),
      bullet("User: marketplace or review service (e-commerce)"),
      bullet("Input: product or service review"),
      bullet("Output: 'suspicious review' flag + confidence score"),
      bullet("Requirement: fast inference, batch-processing capability"),
      spacer(8),

      // 4. Data
      sectionHeading("4. Data"),
      hrRule(TEAL),

      subsectionHeading("4.1  Dataset Sources"),
      bullet("HC3 (Human ChatGPT Comparison Corpus) — question-answer pairs: human and ChatGPT"),
      bullet("TuringBench — benchmark for AI-generated text detection"),
      bullet("RAID Dataset — multi-source dataset from diverse LLMs"),
      bullet("AuTextification (IberLEF 2023) — multilingual, includes Spanish"),
      bullet("Reddit / Wikipedia aggregates — human-written reference text"),
      spacer(6),

      subsectionHeading("4.2  Quality Requirements"),
      bodyText("The supervisor explicitly emphasised: dataset quality is critical — garbage in, garbage out."),
      spacer(4),
      bullet("Class balance: ~50/50 human vs AI (or class weighting)"),
      bullet("Domain diversity: academic texts, news, forums, reviews"),
      bullet("Model diversity: GPT-3.5, GPT-4, LLaMA, Claude — multiple 'AI authors'"),
      bullet("Remove duplicates, noisy records, and unlabelled samples"),
      bullet("Stratified split: train / validation / test (70 / 15 / 15)"),
      spacer(6),

      subsectionHeading("4.3  Preprocessing"),
      bullet("Tokenisation: standard (NLTK / SpaCy) for ML; subword BPE for neural models"),
      bullet("Remove HTML, special characters; normalise whitespace"),
      bullet("For TDA: build embeddings → Vietoris-Rips complexes → compute Betti numbers"),
      spacer(8),

      // 5. Models
      sectionHeading("5. Models and Architectures"),
      hrRule(TEAL),

      bodyText("The project involves a comparative study of five approaches — from a simple lower-bound baseline to topological methods:"),
      spacer(6),
      modelsTable(),
      spacer(8),

      subsectionHeading("5.1  fastText + LSTM — Primary Baseline"),
      bodyText("Recommended by the supervisor as a strong baseline. fastText provides high-quality word embeddings with morphological awareness (important for Russian-language texts). LSTM captures sequential dependencies and local context — key for detecting AI patterns."),
      spacer(4),
      bullet("Architecture: fastText embeddings (dim 300) → LSTM (hidden 256) → Dropout → Dense(2)"),
      bullet("Training: Adam optimizer, lr=1e-3, batch_size=64, early stopping"),
      bullet("Soft labels: sigmoid output + threshold tuning for confidence score"),
      spacer(6),

      subsectionHeading("5.2  GRU vs LSTM — Comparative Experiment"),
      bodyText("Goal: determine which recurrent architecture better models AI-text patterns."),
      spacer(4),
      bullet("GRU: fewer parameters, faster training, sometimes better on short texts"),
      bullet("LSTM: retains longer-range dependencies — preferable for longer documents"),
      bullet("Experiment: identical architecture, same data — only the recurrent cell differs"),
      bullet("Metrics: F1, AUC, training time, parameter count"),
      spacer(6),

      subsectionHeading("5.3  TinyBERT — Advanced Approach"),
      bodyText("Distilled version of BERT — 7.5x smaller and 9x faster while retaining ~97% of BERT's quality. Contextual embeddings are the expected F1 leader."),
      spacer(4),
      bullet("Fine-tuning: huggingface/tiny-bert-for-sequence-classification"),
      bullet("Batch size: 16–32, lr=2e-5, 3–5 epochs, linear warmup"),
      bullet("Enables zero-shot and few-shot detection experiments"),
      spacer(8),

      // 6. Soft labels
      sectionHeading("6. Soft Labels and Confidence Scoring"),
      hrRule(TEAL),

      bodyText("As directed by the supervisor: design a confidence metric instead of hard binary classification. AI texts frequently contain mixed content — human-edited AI output. A soft score is more informative than a binary label."),
      spacer(6),

      subsectionHeading("Proposed Approaches"),
      bullet("Model Confidence Score: softmax probability (P(AI) ∈ [0, 1]) — simple baseline"),
      bullet("Ensemble Agreement: fraction of ensemble models predicting AI — interpretable metric"),
      bullet("Sentence-Level Scoring: score each sentence individually → aggregate (mean, max) → mixed-content detection"),
      bullet("Topological Confidence: normalised perforation metric as a measure of 'AI-likeness'"),
      bullet("Calibrated Probability: temperature scaling to calibrate neural network probabilities"),
      spacer(6),

      bodyText("Final confidence score: weighted combination — model confidence (0.5) + topological score (0.3) + linguistic features (0.2)."),
      spacer(8),

      // 7. Evaluation metrics
      sectionHeading("7. Evaluation Metrics"),
      hrRule(TEAL),
      spacer(4),
      metricsTable(),
      spacer(8),

      // 8. Work plan
      sectionHeading("8. Phased Work Plan"),
      hrRule(TEAL),
      spacer(4),
      phaseBox(1, "Data Collection & EDA", "2–3 weeks", [
        "Collect and merge datasets (HC3, TuringBench, RAID)",
        "Analyse class distributions, domains, and text lengths",
        "Preprocessing: tokenisation, cleaning, normalisation",
        "Stratified train / val / test split",
        "Quality audit: deduplication, noise removal",
      ]),
      spacer(6),
      phaseBox(2, "Baseline: TF-IDF + Classical ML", "1–2 weeks", [
        "TF-IDF vectorisation (unigrams + bigrams)",
        "Train SVM, Logistic Regression, Random Forest",
        "Evaluate Accuracy, F1, AUC — expected low results",
        "Document as confirmed lower bound",
      ]),
      spacer(6),
      phaseBox(3, "Neural Baselines: fastText + LSTM/GRU", "2–3 weeks", [
        "Train fastText embeddings (or use pre-trained)",
        "Build LSTM classifier with dropout and early stopping",
        "Parallel run: identical architecture with GRU cells",
        "GRU vs LSTM comparison: F1, AUC, speed, parameters",
        "Implement confidence score (softmax + threshold tuning)",
      ]),
      spacer(6),
      phaseBox(4, "TinyBERT Fine-tuning", "2 weeks", [
        "Fine-tune TinyBERT on binary classification task",
        "Experiment with learning rate, batch size, epochs",
        "Calibration: temperature scaling for probabilities",
        "Sentence-level scoring for mixed-content detection",
      ]),
      spacer(6),
      phaseBox(5, "Topological Data Analysis (TDA)", "2–3 weeks", [
        "Build embeddings → Vietoris-Rips complexes",
        "Persistent homology: Betti numbers β₀, β₁",
        "Perforation metric following the Hidden Holes paper",
        "Hausdorff dimension + entropy-complexity (Gromov's method)",
        "TDA as standalone classifier + as additional feature for neural models",
      ]),
      spacer(6),
      phaseBox(6, "Comparison & Final Report", "1–2 weeks", [
        "Summary table of all model results",
        "Error analysis (confusion matrix, failure case review)",
        "Implement final ensemble confidence score",
        "Write-up and results presentation",
      ]),
      spacer(8),

      // 9. Tech stack
      sectionHeading("9. Technical Stack"),
      hrRule(TEAL),
      spacer(4),

      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [2400, CONTENT_W - 2400],
        borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideH: noBorder, insideV: noBorder },
        rows: [
          ["Language & Environment", "Python 3.11  ·  Jupyter Notebook  ·  VS Code"],
          ["ML / DL", "scikit-learn  ·  PyTorch  ·  Hugging Face Transformers"],
          ["NLP", "NLTK  ·  SpaCy  ·  fastText  ·  BPE Tokenizer"],
          ["TDA", "Ripser  ·  Gudhi  ·  Persim (persistent homology)"],
          ["Experiment Tracking", "MLflow  ·  Weights & Biases"],
          ["Data", "Pandas  ·  NumPy  ·  HuggingFace datasets"],
          ["Visualisation", "Matplotlib  ·  Seaborn  ·  Plotly"],
          ["Version Control", "Git  ·  GitHub  ·  DVC (data versioning)"],
        ].map(([label, val], i) => new TableRow({
          children: [
            new TableCell({
              borders: noBorders,
              width: { size: 2400, type: WidthType.DXA },
              shading: { fill: i % 2 === 0 ? PALE_TEAL : WHITE, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 120, right: 80 },
              children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 20, color: TEAL, font: "Arial" })] })]
            }),
            new TableCell({
              borders: noBorders,
              width: { size: CONTENT_W - 2400, type: WidthType.DXA },
              shading: { fill: i % 2 === 0 ? PALE_TEAL : WHITE, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 80, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: val, size: 20, color: DARK_NAVY, font: "Arial" })] })]
            }),
          ]
        }))
      }),
      spacer(8),

      // 10. Expected results
      sectionHeading("10. Expected Results"),
      hrRule(TEAL),

      subsectionHeading("Model Performance"),
      bullet("TF-IDF + SVM: Accuracy ~70–75%, F1 ~0.68 — confirmed weak baseline"),
      bullet("fastText + LSTM: Accuracy ~85–90%, F1 ~0.87 — strong recommended baseline"),
      bullet("GRU: comparable to LSTM, potentially faster on short texts"),
      bullet("TinyBERT: Accuracy ~92–95%, F1 ~0.93 — expected leader"),
      bullet("TDA (Gromov-style): Accuracy ~88–93% without neural networks — novel contribution"),
      spacer(4),

      subsectionHeading("Scientific Contribution"),
      bullet("First comparison of TDA methods (Hidden Holes + Gromov) against neural networks on AI detection"),
      bullet("Design and evaluation of an integrated confidence score based on model ensemble"),
      bullet("Reproduction and validation of the perforation metric on new data"),
      bullet("Practical guidance on approach selection for different deployment scenarios"),
      spacer(8),

      // 11. Risks
      sectionHeading("11. Risks and Limitations"),
      hrRule(TEAL),

      bullet("Data drift: models trained on specific LLMs — newer GPT/Claude versions may evade the detector"),
      bullet("Compute requirements: TinyBERT requires GPU; TDA has high complexity for long texts"),
      bullet("Dataset quality: difficulty verifying that 'human' text was not AI-assisted"),
      bullet("Multilingual scope: most datasets are English; Russian-language texts require separate treatment"),
      bullet("Adversarial robustness: paraphrasing and AI-style masking — system resilience untested"),
      spacer(8),

      // References
      sectionHeading("Key References"),
      hrRule(TEAL),

      ...[
        ["[1]", "Kushnareva et al. \"Artificial Text Boundary Detection with Topological Data Analysis\" — Hidden Holes paper. Persistent homology for AI text detection."],
        ["[2]", "Gromov A. \"Spot the Bot: Language as a Self-Organised Critical System\" — HSE University. Hausdorff dimension, entropy-complexity, TDA without neural networks (93–97%)."],
        ["[3]", "Guo et al. \"How Close is ChatGPT to Human Experts?\" — HC3 dataset, ChatGPT detection baseline."],
        ["[4]", "Jia et al. \"TinyBERT: Distilling BERT for Natural Language Understanding\" — TinyBERT architecture."],
        ["[5]", "Boyarkov et al. \"RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors\" — 2023."],
      ].map(([ref, text]) => new Paragraph({
        spacing: { before: 50, after: 40 },
        children: [
          new TextRun({ text: ref + "  ", bold: true, size: 20, color: TEAL, font: "Arial" }),
          new TextRun({ text, size: 20, color: DARK_NAVY, font: "Arial" }),
        ]
      })),

      spacer(6),
      hrRule(LIGHT_GREY),
      new Paragraph({
        spacing: { before: 60, after: 0 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Nabil Mouhamech  ·  HSE University  ·  Master's in Applied AI  ·  2026", size: 18, color: MID_GREY, font: "Arial" })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("AI_Detection_Project_Plan_EN.docx", buffer);
  console.log("AI_Detection_Project_Plan_EN.docx created successfully");
});
