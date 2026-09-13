# Interview Guide — NLP Customer Support Intent Routing

## 60-second walkthrough
I built a customer-support routing system around the BANKING77 dataset. The project compares a TF-IDF logistic baseline, BiLSTM, and DistilBERT, then turns model confidence into an operational policy: auto-route only high-confidence requests and escalate ambiguous cases to humans. Evaluation emphasizes macro-F1, class-level error patterns, calibration, automated coverage, and the quality of auto-routed cases—not accuracy alone.

## Know these ideas
- **Macro-F1:** gives each intent equal weight, which matters across 77 classes.
- **Baseline value:** TF-IDF + logistic shows whether more expensive representations add real value.
- **Calibration:** predicted confidence should correspond to observed correctness.
- **Selective prediction:** automate only when confidence clears a validated quality constraint.
- **Confusion analysis:** repeated intent confusions may reflect labeling taxonomy, training data, or model limitations.

## Likely questions
**Why not route every query automatically?**  
Wrong routing creates customer friction and delays resolution. A confidence-based review policy makes the quality/automation trade-off explicit.

**What would you monitor after launch?**  
Auto-routing coverage, accuracy on auto-routed cases, human-review rate, per-intent error shifts, confidence calibration, and emerging unknown intents.

## Reproduce and learn
1. Explain the three-model sequence and the cost/quality trade-off.
2. Read the generated results after the full benchmark workflow runs.
3. Inspect a top confusion pair and propose a fix.
4. State how the threshold is selected.
5. Deliver the walkthrough without claiming unverified benchmark numbers.

## Honest boundary
The full benchmark workflow produces the final comparison artifacts; resume metrics must come only from those verified outputs.