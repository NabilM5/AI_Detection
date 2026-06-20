const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, TabStopType, LevelFormat, PageBreak, ExternalHyperlink
} = require('docx');
const fs = require('fs');

// Document theme
const TEAL       = "007A8A";
const DARK_NAVY  = "1A2A3A";
const MID_GREY   = "555555";
const LIGHT_GREY = "EEEEEE";
const PALE_TEAL  = "E6F4F6";
const WHITE      = "FFFFFF";
const ORANGE     = "C0620C";

// A4 layout
const PAGE_W    = 11906;
const PAGE_H    = 16838;
const MARGIN    = 1134; // ~2cm
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

function bullet(text, bold = false) {
  return new Paragraph({
    spacing: { before: 30, after: 30 },
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, size: 22, color: DARK_NAVY, font: "Arial", bold })]
  });
}

function subBullet(text) {
  return new Paragraph({
    spacing: { before: 20, after: 20 },
    numbering: { reference: "subbullets", level: 0 },
    children: [new TextRun({ text, size: 21, color: MID_GREY, font: "Arial" })]
  });
}

function numberedItem(text, num) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    numbering: { reference: "numbers", level: 0 },
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

function phaseBox(phaseNum, title, duration, description, tasks) {
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
    rows: [
      new TableRow({ children: [headerCell] }),
      ...bodyRows
    ]
  });
}

function metricsTable() {
  const headers = ["Метрика", "Описание", "Почему важна"];
  const rows = [
    ["Accuracy", "Доля верных предсказаний", "Базовый показатель"],
    ["F1-Score", "Гармоническое среднее точности и полноты", "Учитывает дисбаланс классов"],
    ["ROC-AUC", "Площадь под кривой ROC", "Устойчивость к порогу классификации"],
    ["Confidence Score", "Уверенность модели в предсказании (0–1)", "Мягкие метки — ключевая задача"],
    ["Topological Complexity", "Число дыр (Betti-числа) в данных", "Distinguishes human vs AI structure"],
    ["Hausdorff Dimension", "Фрактальная размерность текста", "Критерий самоорганизованной критичности"],
  ];

  const colWidths = [1800, 3800, CONTENT_W - 5600];
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

  const dataRows = rows.map(row => new TableRow({
    children: row.map((cell, i) => new TableCell({
      borders: thinBorders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: i === 0 ? PALE_TEAL : WHITE, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, color: DARK_NAVY, font: "Arial", bold: i === 0 })] })]
    }))
  }));

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [header, ...dataRows]
  });
}

function modelsTable() {
  const colWidths = [2200, 2600, 2000, CONTENT_W - 6800];
  const headers = ["Модель", "Тип", "Сложность", "Обоснование"];
  const rows = [
    ["TF-IDF + SVM", "Classical ML (baseline)", "Низкая", "Стандартный baseline; логически не даст высоких результатов (замечание преподавателя)"],
    ["fastText + LSTM", "Гибридный (слова + посл.)", "Средняя", "Рекомендован преподавателем как сильный baseline — учитывает локальный контекст"],
    ["GRU vs LSTM", "Сравнение RNN-архитектур", "Средняя", "Эксперимент: какая рекуррентная сеть лучше улавливает AI-паттерны?"],
    ["TinyBERT", "Transformer (дистилляция)", "Высокая", "Контекстуальные эмбеддинги; ожидаемый лидер по F1"],
    ["TDA Pipeline", "Топологический анализ", "Экспер.", "Perforation metric + Betti numbers (Hidden Holes paper)"],
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
      children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, color: i === 2 && cell === "Высокая" ? "8B0000" : DARK_NAVY, font: "Arial", bold: i === 0 })] })]
    }))
  }));

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [header, ...dataRows]
  });
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
        reference: "subbullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 800, hanging: 280 } } } }]
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
        children: [new TextRun({ text: "ПЛАН ИССЛЕДОВАТЕЛЬСКОГО ПРОЕКТА", size: 20, color: MID_GREY, font: "Arial", bold: true })]
      }),
      new Paragraph({
        spacing: { before: 0, after: 40 },
        children: [new TextRun({ text: "Семинарский проект · НИУ ВШЭ · 2026", size: 20, color: MID_GREY, font: "Arial" })]
      }),
      hrRule(TEAL),
      new Paragraph({
        spacing: { before: 80, after: 60 },
        children: [new TextRun({ text: "Обнаружение текстов,", bold: true, size: 52, color: DARK_NAVY, font: "Arial" })]
      }),
      new Paragraph({
        spacing: { before: 0, after: 60 },
        children: [new TextRun({ text: "сгенерированных ИИ", bold: true, size: 52, color: DARK_NAVY, font: "Arial" })]
      }),
      new Paragraph({
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Detection of AI-Generated Text", size: 28, color: TEAL, font: "Arial" })]
      }),
      hrRule(LIGHT_GREY),
      spacer(6),
      infoBox("Автор", "Набиль Мухамеш  ·  Магистратура «Прикладной ИИ»"),
      spacer(4),
      infoBox("Университет", "НИУ ВШЭ — Высшая школа экономики · Москва"),
      spacer(4),
      infoBox("Тип работы", "Семинарский исследовательский проект"),
      spacer(4),
      infoBox("Статус", "Планирование / In Progress"),
      spacer(4),
      infoBox("Дата", "2026"),
      spacer(12),

      // 1. Обоснование
      sectionHeading("1. Обоснование и актуальность"),
      hrRule(TEAL),

      bodyText("Широкое распространение крупных языковых моделей (LLM) — GPT-4, LLaMA, Gemini — создаёт растущую угрозу злоупотреблений: академический плагиат, дезинформация, поддельный контент. Автоматическое обнаружение AI-сгенерированных текстов становится критически важной задачей."),
      spacer(4),
      bodyText("Исследование сравнивает классические ML-методы и нейросетевые подходы для бинарной классификации: человеческий текст vs AI-сгенерированный. Помимо этого, проект исследует возможности топологического анализа данных (TDA) как альтернативного подхода, основанного на структурных свойствах языка."),
      spacer(4),
      subsectionHeading("Ключевые исследовательские вопросы"),
      bullet("Какой подход — классический ML (TF-IDF + SVM) или нейросетевой (fastText+LSTM, TinyBERT) — даёт наилучшие результаты?"),
      bullet("Чем различается топологическая сложность AI-текстов и текстов, написанных людьми?"),
      bullet("Применима ли «перфорационная метрика» (Hidden Holes paper) для детектирования AI?"),
      bullet("Как GRU и LSTM ведут себя на задаче детектирования — что лучше улавливает AI-паттерны?"),
      bullet("Можно ли генерировать мягкие метки (soft labels) — оценку уверенности вместо бинарного решения?"),
      spacer(8),

      // 2. Теоретическая база
      sectionHeading("2. Теоретическая база"),
      hrRule(TEAL),

      subsectionHeading("2.1  «Hidden Holes» — Топологический анализ текстов"),
      bodyText("Статья (Kushnareva et al.) демонстрирует, что тексты, написанные людьми, обладают более высокой топологической сложностью, чем AI-сгенерированные. Для измерения этого используется устойчивая гомология (persistent homology) и комплексы Вьеториса-Рипса."),
      spacer(4),
      bullet("Перфорационная метрика («perforation metric»): измеряет число и устойчивость топологических дыр (Betti-числа β₁) в пространстве высокоразмерных эмбеддингов"),
      bullet("Ключевой результат: LSTM >> Transformer с точки зрения топологической сложности, что объясняет, почему LSTM-тексты труднее отличить от человеческих"),
      bullet("Практическое применение: вычисление перфорационной метрики как дополнительного признака для классификатора"),
      spacer(8),

      subsectionHeading("2.2  «Spot the Bot» — Метод Громова (HSE)"),
      bodyText("Громов рассматривает язык как самоорганизованную критическую систему — аналог физических систем на пороге фазового перехода. AI-тексты нарушают эту критичность, что проявляется в измеримых статистических аномалиях."),
      spacer(4),
      bullet("Размерность Хаусдорфа: фрактальная характеристика текста; у AI-текстов она систематически отличается от человеческих"),
      bullet("Энтропия-сложность (entropy-complexity plane): AI-тексты концентрируются в специфической области этого пространства"),
      bullet("TDA без нейросетей: метод достигает 93–97% точности исключительно на топологических признаках — без обучения нейросети"),
      bullet("Диагностический потенциал: система выявляет стилометрические отклонения и автора-«бота»"),
      spacer(8),

      subsectionHeading("2.3  Рекомендации преподавателя"),
      bodyText("Преподаватель НИУ ВШЭ дал следующие конкретные указания по развитию проекта:"),
      spacer(4),
      bullet("Пользовательский сценарий: описать конкретные use-cases — студенческие работы, новостные статьи, отзывы"),
      bullet("Качество датасета: обратить особое внимание на качество обучающих данных — garbage in, garbage out"),
      bullet("Статистические и топологические методы: применить TDA наряду с нейросетями"),
      bullet("Baseline: fastText + LSTM — рекомендован как сильный baseline, учитывающий локальный контекст"),
      bullet("Сравнение архитектур: GRU vs LSTM — провести эксперимент"),
      bullet("Продвинутая модель: попробовать TinyBERT — дистиллированный BERT"),
      bullet("Мягкие метки: изобрести метрику уверенности вместо жёсткой бинарной классификации"),
      bullet("TF-IDF: логически не даст высокого результата — включить как низший baseline для подтверждения"),
      spacer(8),

      // 3. Пользовательские сценарии
      sectionHeading("3. Пользовательские сценарии"),
      hrRule(TEAL),

      bodyText("Согласно указанию преподавателя, необходимо чётко определить, кто будет использовать систему и в каких условиях."),
      spacer(6),

      subsectionHeading("Сценарий A: Академическая среда"),
      bullet("Пользователь: преподаватель, ассистент кафедры, антиплагиатная система"),
      bullet("Вход: студенческое эссе, курсовая или семинарская работа"),
      bullet("Выход: вероятность AI-генерации + highlighted фрагменты с высокой уверенностью"),
      bullet("Требования: высокий recall (лучше ложная тревога, чем пропуск AI-текста)"),
      spacer(6),

      subsectionHeading("Сценарий B: Медиа и верификация фактов"),
      bullet("Пользователь: редактор, fact-checker, платформа публикации"),
      bullet("Вход: новостная статья, пресс-релиз, пост в социальных сетях"),
      bullet("Выход: оценка достоверности + источник (human / AI / mixed)"),
      bullet("Требования: высокая precision, объяснимость решения"),
      spacer(6),

      subsectionHeading("Сценарий C: Платформы отзывов"),
      bullet("Пользователь: маркетплейс, сервис отзывов (e-commerce)"),
      bullet("Вход: отзыв на товар или услугу"),
      bullet("Выход: флаг «подозрительный отзыв» + confidence score"),
      bullet("Требования: скорость работы, возможность batch-обработки"),
      spacer(8),

      // 4. Данные
      sectionHeading("4. Данные"),
      hrRule(TEAL),

      subsectionHeading("4.1  Источники датасетов"),
      bullet("HC3 (Human ChatGPT Comparison Corpus) — пары вопрос-ответ: человек и ChatGPT"),
      bullet("TuringBench — бенчмарк для детектирования AI-текстов"),
      bullet("RAID Dataset — многоисточниковый датасет разных LLM"),
      bullet("AuTextification (IberLEF 2023) — многоязычный, включая испанский"),
      bullet("Сборные данные Reddit / Wikipedia — человеческий текст-эталон"),
      spacer(6),

      subsectionHeading("4.2  Требования к качеству"),
      bodyText("Преподаватель особо подчеркнул: качество датасета критично — «garbage in, garbage out»."),
      spacer(4),
      bullet("Баланс классов: ~50/50 human vs AI (или применение взвешивания классов)"),
      bullet("Разнообразие доменов: академические тексты, новости, форумы, отзывы"),
      bullet("Разнообразие моделей: GPT-3.5, GPT-4, LLaMA, Claude — разные «авторы»"),
      bullet("Удаление дубликатов, шумовых записей, неразмеченных образцов"),
      bullet("Стратифицированное разбиение: train / validation / test (70 / 15 / 15)"),
      spacer(6),

      subsectionHeading("4.3  Предобработка"),
      bullet("Токенизация: стандартная (NLTK / SpaCy) для ML, subword tokenization (BPE) для нейросетей"),
      bullet("Удаление HTML, спецсимволов, нормализация пробелов"),
      bullet("Для TDA: построение эмбеддингов → Vietoris-Rips комплексы → вычисление Betti-чисел"),
      spacer(8),

      // 5. Модели
      sectionHeading("5. Модели и архитектуры"),
      hrRule(TEAL),

      bodyText("Проект включает сравнительный анализ пяти подходов — от простого baseline до топологических методов:"),
      spacer(6),
      modelsTable(),
      spacer(8),

      subsectionHeading("5.1  fastText + LSTM — основной baseline"),
      bodyText("Рекомендован преподавателем как сильный baseline. fastText даёт качественные word embeddings с учётом морфологии (важно для русскоязычных текстов). LSTM улавливает последовательные зависимости и локальный контекст — ключевые для детектирования AI-паттернов."),
      spacer(4),
      bullet("Входные данные: fastText embeddings (dim 300) + LSTM (hidden 256) + Dropout + Dense(2)"),
      bullet("Обучение: Adam optimizer, lr=1e-3, batch_size=64, early stopping"),
      bullet("Soft labels: выходной слой sigmoid + threshold tuning для confidence score"),
      spacer(6),

      subsectionHeading("5.2  GRU vs LSTM — сравнительный эксперимент"),
      bodyText("Цель: выяснить, какая рекуррентная архитектура лучше моделирует AI-паттерны в тексте."),
      spacer(4),
      bullet("GRU: меньше параметров, быстрее обучается, иногда лучше работает на коротких текстах"),
      bullet("LSTM: дольше удерживает долгосрочные зависимости — предпочтительно для длинных текстов"),
      bullet("Эксперимент: идентичная архитектура, одинаковые данные — только GRU vs LSTM ячейки"),
      bullet("Метрики: F1, AUC, время обучения, количество параметров"),
      spacer(6),

      subsectionHeading("5.3  TinyBERT — продвинутый подход"),
      bodyText("Дистиллированная версия BERT — в 7.5 раз меньше и в 9 раз быстрее, при сохранении ~97% качества. Контекстуальные эмбеддинги — ожидаемый лидер по F1."),
      spacer(4),
      bullet("Fine-tuning: huggingface/tiny-bert-for-sequence-classification"),
      bullet("Batch size: 16–32, lr=2e-5, 3–5 эпох, linear warmup"),
      bullet("Возможность zero-shot и few-shot детектирования"),
      spacer(8),

      // 6. Мягкие метки
      sectionHeading("6. Мягкие метки и метрика уверенности"),
      hrRule(TEAL),

      bodyText("По указанию преподавателя: необходимо «изобрести» метрику уверенности вместо жёсткой бинарной классификации. AI-тексты часто содержат смешанный контент — человек редактировал AI-вывод. Мягкая оценка информативнее бинарной метки."),
      spacer(6),

      subsectionHeading("Предлагаемые подходы"),
      bullet("Model Confidence Score: вероятность из softmax (P(AI) ∈ [0, 1]) — простой baseline"),
      bullet("Ensemble Agreement: доля моделей ансамбля, предсказавших AI — интерпретируемая метрика"),
      bullet("Sentence-Level Score: оценка каждого предложения отдельно → агрегация (mean, max) → mixed-content detection"),
      bullet("Topological Confidence: нормированная перфорационная метрика как мера «похожести на AI»"),
      bullet("Calibrated Probability: temperature scaling для калибровки вероятностей нейросети"),
      spacer(6),

      bodyText("Итоговый confidence score — взвешенная комбинация: модельная уверенность (0.5) + топологический показатель (0.3) + лингвистические признаки (0.2)."),
      spacer(8),

      // 7. Метрики
      sectionHeading("7. Метрики оценки"),
      hrRule(TEAL),
      spacer(4),
      metricsTable(),
      spacer(8),

      // 8. План работ
      sectionHeading("8. План работ по фазам"),
      hrRule(TEAL),
      spacer(4),
      phaseBox(1, "Подготовка данных и EDA", "2–3 недели", "", [
        "Сбор и объединение датасетов (HC3, TuringBench, RAID)",
        "Анализ распределения классов, доменов, длин текстов",
        "Предобработка: токенизация, очистка, нормализация",
        "Стратифицированное разбиение train/val/test",
        "Анализ качества: удаление дубликатов, шумовых примеров",
      ]),
      spacer(6),
      phaseBox(2, "Baseline: TF-IDF + Classical ML", "1–2 недели", "", [
        "Векторизация TF-IDF (unigrams + bigrams)",
        "Обучение SVM, Logistic Regression, Random Forest",
        "Оценка Accuracy, F1, AUC — ожидаемо невысокие результаты",
        "Документирование как нижней границы качества",
      ]),
      spacer(6),
      phaseBox(3, "Neural Baselines: fastText + LSTM/GRU", "2–3 недели", "", [
        "Обучение fastText embeddings на данных (или предобученные)",
        "Построение LSTM-классификатора с dropout и early stopping",
        "Параллельно: идентичная архитектура на GRU",
        "Сравнение GRU vs LSTM: F1, AUC, время, параметры",
        "Реализация confidence score (softmax + threshold tuning)",
      ]),
      spacer(6),
      phaseBox(4, "TinyBERT Fine-tuning", "2 недели", "", [
        "Fine-tuning TinyBERT на задаче бинарной классификации",
        "Эксперименты с learning rate, batch size, эпохами",
        "Calibration: temperature scaling для вероятностей",
        "Sentence-level scoring для mixed-content detection",
      ]),
      spacer(6),
      phaseBox(5, "Топологический анализ (TDA)", "2–3 недели", "", [
        "Вычисление эмбеддингов → Vietoris-Rips комплексы",
        "Persistent homology: Betti numbers β₀, β₁",
        "Перфорационная метрика по Hidden Holes paper",
        "Размерность Хаусдорфа + entropy-complexity (Gromov)",
        "TDA как standalone классификатор + как дополнительный признак",
      ]),
      spacer(6),
      phaseBox(6, "Сравнение и финальный отчёт", "1–2 недели", "", [
        "Сводная таблица результатов всех моделей",
        "Анализ ошибок (confusion matrix, error analysis)",
        "Реализация финального confidence score (ensemble)",
        "Написание отчёта + презентация результатов",
      ]),
      spacer(8),

      // 9. Технический стек
      sectionHeading("9. Технический стек"),
      hrRule(TEAL),
      spacer(4),

      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [2400, CONTENT_W - 2400],
        borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideH: noBorder, insideV: noBorder },
        rows: [
          ["Язык и среда", "Python 3.11 · Jupyter Notebook · VS Code"],
          ["ML / DL", "scikit-learn · PyTorch · Hugging Face Transformers"],
          ["NLP", "NLTK · SpaCy · fastText · BPE Tokenizer"],
          ["TDA", "Ripser · Gudhi · Persim (persistent homology)"],
          ["Эксперименты", "MLflow · Weights & Biases (tracking)"],
          ["Данные", "Pandas · NumPy · datasets (HuggingFace)"],
          ["Визуализация", "Matplotlib · Seaborn · Plotly"],
          ["Версии", "Git · GitHub · DVC (data versioning)"],
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

      // 10. Ожидаемые результаты
      sectionHeading("10. Ожидаемые результаты"),
      hrRule(TEAL),

      subsectionHeading("По качеству моделей"),
      bullet("TF-IDF + SVM: Accuracy ~70–75%, F1 ~0.68 — подтверждённо слабый baseline"),
      bullet("fastText + LSTM: Accuracy ~85–90%, F1 ~0.87 — сильный рекомендованный baseline"),
      bullet("GRU: сопоставим с LSTM, потенциально быстрее на коротких текстах"),
      bullet("TinyBERT: Accuracy ~92–95%, F1 ~0.93 — ожидаемый лидер"),
      bullet("TDA (Gromov-style): Accuracy ~88–93% без нейросетей — новаторский результат"),
      spacer(4),

      subsectionHeading("Научный вклад"),
      bullet("Первое сравнение TDA-методов (Hidden Holes + Gromov) с нейросетями на задаче AI-detection"),
      bullet("Разработка интегрированного confidence score на основе ансамбля"),
      bullet("Воспроизведение и проверка перфорационной метрики на новых данных"),
      bullet("Практические рекомендации по выбору подхода для разных сценариев"),
      spacer(8),

      // 11. Риски
      sectionHeading("11. Риски и ограничения"),
      hrRule(TEAL),

      bullet("Дрейф данных: модели обучены на конкретных LLM — новые версии GPT/Claude могут обойти детектор"),
      bullet("Вычислительные ресурсы: TinyBERT требует GPU; TDA — высокая сложность для длинных текстов"),
      bullet("Качество датасетов: сложность верификации, что «человеческий» текст не отредактирован AI"),
      bullet("Мультиязычность: большинство датасетов на английском; русскоязычные тексты — отдельная задача"),
      bullet("Adversarial атаки: паrafhrasing, маскировка AI-стиля — устойчивость системы под вопросом"),
      spacer(8),

      // Литература
      sectionHeading("Ключевые источники"),
      hrRule(TEAL),

      new Paragraph({
        spacing: { before: 60, after: 40 },
        children: [
          new TextRun({ text: "[1]  ", bold: true, size: 20, color: TEAL, font: "Arial" }),
          new TextRun({ text: "Kushnareva et al. «Artificial Text Boundary Detection with Topological Data Analysis» — Hidden Holes paper. Персистентная гомология для обнаружения AI-текстов.", size: 20, color: DARK_NAVY, font: "Arial" }),
        ]
      }),
      new Paragraph({
        spacing: { before: 40, after: 40 },
        children: [
          new TextRun({ text: "[2]  ", bold: true, size: 20, color: TEAL, font: "Arial" }),
          new TextRun({ text: "Gromov A. «Spot the Bot: Language as a Self-Organized Critical System» — НИУ ВШЭ. Размерность Хаусдорфа, энтропия-сложность, TDA без нейросетей (93–97%).", size: 20, color: DARK_NAVY, font: "Arial" }),
        ]
      }),
      new Paragraph({
        spacing: { before: 40, after: 40 },
        children: [
          new TextRun({ text: "[3]  ", bold: true, size: 20, color: TEAL, font: "Arial" }),
          new TextRun({ text: "Guo et al. «How Close is ChatGPT to Human Experts?» — HC3 датасет, baseline для детектирования ChatGPT.", size: 20, color: DARK_NAVY, font: "Arial" }),
        ]
      }),
      new Paragraph({
        spacing: { before: 40, after: 40 },
        children: [
          new TextRun({ text: "[4]  ", bold: true, size: 20, color: TEAL, font: "Arial" }),
          new TextRun({ text: "Jia et al. «TinyBERT: Distilling BERT for Natural Language Understanding» — архитектура TinyBERT.", size: 20, color: DARK_NAVY, font: "Arial" }),
        ]
      }),
      new Paragraph({
        spacing: { before: 40, after: 80 },
        children: [
          new TextRun({ text: "[5]  ", bold: true, size: 20, color: TEAL, font: "Arial" }),
          new TextRun({ text: "Боянков и др. «RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors» — 2023.", size: 20, color: DARK_NAVY, font: "Arial" }),
        ]
      }),

      hrRule(LIGHT_GREY),
      new Paragraph({
        spacing: { before: 60, after: 0 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Набиль Мухамеш  ·  НИУ ВШЭ  ·  Магистратура «Прикладной ИИ»  ·  2026", size: 18, color: MID_GREY, font: "Arial" })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("AI_Detection_Project_Plan.docx", buffer);
  console.log("AI_Detection_Project_Plan.docx created successfully");
});
