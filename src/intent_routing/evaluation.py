import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

def per_class_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}:
            rows.append({"label": label, **values})
    return pd.DataFrame(rows)

def top_confusions(y_true, y_pred, labels, top_n=20):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                rows.append({"true_label": true_label, "pred_label": pred_label, "count": int(cm[i, j])})
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n).reset_index(drop=True)

def expected_calibration_error(y_true, probs, classes, bins=10):
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    correct = np.asarray(pred == np.asarray(y_true), dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
