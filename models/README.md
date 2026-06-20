# Models

Trained model artifacts are generated locally and intentionally excluded from Git.

Training scripts write artifacts such as:

```text
models/
├── tfidf_logreg.pkl
├── tfidf_vectorizer.pkl
├── lstm_baseline.pt
├── lstm_vocab.pkl
├── gru_baseline.pt
├── gru_vocab.pkl
├── pretrained_embedding_lstm.pt
├── pretrained_embedding_lstm_vocab.pkl
├── tinybert_baseline/
├── cross_source_tfidf/
└── statistical_classifier/
```

Vocabulary files must be kept with their corresponding LSTM/GRU weights because they define the token-to-ID mapping used during training.

