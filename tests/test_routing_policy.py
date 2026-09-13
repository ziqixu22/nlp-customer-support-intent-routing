import numpy as np
from intent_routing.routing import evaluate_routing, select_threshold

def test_confident_examples_are_auto_routed():
    classes = np.array(["alpha", "beta"])
    truth = np.array(["alpha", "beta", "alpha", "beta"])
    probs = np.array([[0.9,0.1],[0.2,0.8],[0.55,0.45],[0.45,0.55]])
    result = evaluate_routing(truth, probs, classes, 0.7)
    assert result.auto_coverage == 0.5
    assert result.auto_accuracy == 1.0
    assert result.review_rate == 0.5

def test_threshold_is_selected_on_accuracy_constraint():
    classes = np.array(["alpha", "beta"])
    truth = np.array(["alpha", "beta", "alpha", "beta", "alpha"])
    probs = np.array([[0.95,0.05],[0.05,0.95],[0.8,0.2],[0.6,0.4],[0.51,0.49]])
    result = select_threshold(truth, probs, classes, min_auto_accuracy=0.9)
    assert result.auto_accuracy >= 0.9
