import pandas as pd
from intent_routing.data import make_train_valid

def test_stratified_validation_contains_every_label():
    frame = pd.DataFrame({
        "text": [str(i) for i in range(40)],
        "label": ["alpha"] * 20 + ["beta"] * 20,
    })
    train, valid = make_train_valid(frame, valid_size=0.2, seed=42)
    assert set(train["label"]) == {"alpha", "beta"}
    assert set(valid["label"]) == {"alpha", "beta"}
    assert len(train) + len(valid) == len(frame)
