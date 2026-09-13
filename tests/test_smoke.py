from intent_routing.baseline import build_tfidf_logreg
from intent_routing.routing import route_by_confidence

def test_baseline_builds():
    assert build_tfidf_logreg() is not None

def test_routing_shape():
    import numpy as np
    mask = route_by_confidence(np.array([[0.9, 0.1], [0.5, 0.5]]), 0.7)
    assert mask.tolist() == [True, False]
