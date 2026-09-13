# Learning Guide — NLP Customer Support Intent Routing

## Business problem
Route short banking support messages to one of 77 intents while sending ambiguous cases to human review.

## Data discipline
Use the official BANKING77 test split only for final evaluation. Create a stratified validation split from training data for model selection and routing-threshold selection.

## Model 1: TF-IDF + Logistic Regression
TF-IDF creates sparse lexical features from unigrams and bigrams. Logistic Regression is a strong, fast multiclass baseline for short text. It establishes whether heavier neural models add enough value.

## Model 2: TensorFlow BiLSTM
Text is mapped to token ids, dense embeddings, and a bidirectional LSTM. This benchmark learns task-specific sequence representations rather than fixed sparse lexical features.

## Model 3: PyTorch + DistilBERT
A pretrained Transformer is fine-tuned for 77-way intent classification. Learn subword tokenization, attention masks, contextual embeddings, self-attention, transfer learning, fine-tuning, learning rate, epochs, and overfitting control.

## Evaluation
Use accuracy plus macro-F1, per-class precision/recall/F1, confusion pairs, and confidence diagnostics. Macro-F1 gives each intent equal weight and prevents easy classes from hiding weak classes.

## Routing layer
For each prediction, use the maximum class probability as confidence. Select a threshold on validation data that maximizes auto-routing coverage subject to a target accuracy for auto-routed cases. Send lower-confidence cases to human review.

## Deep-dive questions
Be able to explain why a baseline matters, how TF-IDF works, why bigrams help, why macro-F1 matters, what embeddings represent, how an LSTM differs from a Transformer, query/key/value intuition, why DistilBERT is fine-tuned instead of trained from scratch, why test data cannot select thresholds, why softmax confidence can be miscalibrated, and how intent drift would be monitored.

## Production limitations
BANKING77 is an offline public dataset. A real deployment would require calibrated probabilities, latency and cost measurement, human-review outcome data, taxonomy/version management, and drift monitoring.
