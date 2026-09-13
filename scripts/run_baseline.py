import argparse, json
from pathlib import Path
import joblib
from intent_routing.data import prepare_splits
from intent_routing.baseline import build_tfidf_logreg
from intent_routing.evaluation import classification_metrics, per_class_report, top_confusions, expected_calibration_error
from intent_routing.routing import select_threshold, evaluate_routing

def main(train_path, test_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = prepare_splits(train_path, test_path)
    model = build_tfidf_logreg(); model.fit(s.train.text, s.train.label)
    classes = model.named_steps["clf"].classes_
    vp = model.predict_proba(s.valid.text); tp = model.predict_proba(s.test.text)
    vpred = classes[vp.argmax(1)]; tpred = classes[tp.argmax(1)]
    route = select_threshold(s.valid.label.to_numpy(), vp, classes, .90)
    test_route = evaluate_routing(s.test.label.to_numpy(), tp, classes, route.threshold)
    result = {
        "model": "tfidf_logistic_regression",
        "train_n": len(s.train), "valid_n": len(s.valid), "test_n": len(s.test),
        "validation": classification_metrics(s.valid.label, vpred),
        "test": classification_metrics(s.test.label, tpred),
        "validation_ece": expected_calibration_error(s.valid.label.to_numpy(), vp, classes),
        "test_ece": expected_calibration_error(s.test.label.to_numpy(), tp, classes),
        "selected_validation_routing": route.to_dict(),
        "test_routing": test_route.to_dict(),
    }
    per_class_report(s.test.label, tpred).to_csv(out/"baseline_per_class.csv", index=False)
    top_confusions(s.test.label, tpred, list(classes)).to_csv(out/"baseline_top_confusions.csv", index=False)
    (out/"baseline_metrics.json").write_text(json.dumps(result, indent=2))
    joblib.dump(model, out/"baseline.joblib")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", default="results")
    a = ap.parse_args(); main(a.train, a.test, a.out)
