from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class RoutingMetrics:
    threshold: float
    auto_coverage: float
    auto_accuracy: float
    review_rate: float
    overall_accuracy_if_review_perfect: float
    def to_dict(self):
        return asdict(self)

def route_by_confidence(probs, threshold):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D matrix")
    return probs.max(axis=1) >= threshold

def evaluate_routing(y_true, probs, classes, threshold):
    y = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    pred = classes[probs.argmax(axis=1)]
    auto = route_by_confidence(probs, threshold)
    coverage = float(auto.mean())
    auto_accuracy = float((pred[auto] == y[auto]).mean()) if auto.any() else float("nan")
    overall = float(((pred == y) & auto).sum() + (~auto).sum()) / len(y)
    return RoutingMetrics(float(threshold), coverage, auto_accuracy, float(1-coverage), overall)

def select_threshold(y_true, probs, classes, min_auto_accuracy=0.90):
    candidates = np.linspace(0.30, 0.95, 66)
    feasible = []
    for t in candidates:
        r = evaluate_routing(y_true, probs, classes, float(t))
        if not np.isnan(r.auto_accuracy) and r.auto_accuracy >= min_auto_accuracy:
            feasible.append(r)
    if feasible:
        return max(feasible, key=lambda r: r.auto_coverage)
    return max((evaluate_routing(y_true, probs, classes, float(t)) for t in candidates), key=lambda r: -1 if np.isnan(r.auto_accuracy) else r.auto_accuracy)
