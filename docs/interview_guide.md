# Interview Guide — NLP Customer Support Intent Routing

## 30-second version
I built an NLP routing system for fine-grained banking support queries. I started with a TF-IDF plus Logistic Regression baseline, added a TensorFlow BiLSTM benchmark, and fine-tuned DistilBERT with PyTorch and Hugging Face. I evaluated aggregate and class-level quality across 77 intents, then selected a confidence threshold on validation data so high-confidence cases could be auto-routed while ambiguous cases were escalated to human review.

## Why not jump directly to BERT?
A simple baseline measures how much contextual modeling actually adds. If lexical features already perform well, a Transformer may not justify extra latency, memory, training, and serving cost.

## Why macro-F1?
Each intent matters operationally. Macro-F1 gives every class equal weight instead of allowing frequent or easy classes to dominate the headline metric.

## Why confidence routing?
A wrong automatic route creates rework and customer friction. Human escalation turns uncertainty into an explicit operational decision.

## Why use validation for the threshold?
Choosing the routing threshold on test data would leak information and overstate expected generalization.

## Why DistilBERT?
It gives pretrained contextual language understanding with lower inference cost than larger BERT-family models, making it a reasonable production-oriented candidate for short support messages.

## Limitations
The dataset is offline and single-domain. Real production work would need probability calibration, latency/cost benchmarks, intent-drift monitoring, human-review outcomes, and taxonomy maintenance.
