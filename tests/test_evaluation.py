import numpy as np
from intent_routing.evaluation import classification_metrics, expected_calibration_error

def test_perfect_predictions_score_one():
    truth = ["one", "two", "one"]
    metrics = classification_metrics(truth, truth)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0

def test_calibration_error_is_nonnegative():
    truth = np.array(["one", "two"])
    classes = np.array(["one", "two"])
    probs = np.array([[0.9, 0.1], [0.4, 0.6]])
    assert expected_calibration_error(truth, probs, classes) >= 0.0
