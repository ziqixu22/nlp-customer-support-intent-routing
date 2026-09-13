import argparse, json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from intent_routing.data import prepare_splits
from intent_routing.tensorflow_model import build_bilstm
from intent_routing.evaluation import classification_metrics
from intent_routing.routing import select_threshold, evaluate_routing

def main(train_path, test_path, out_dir):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = prepare_splits(train_path, test_path)
    le = LabelEncoder().fit(s.train.label)
    ytr = le.transform(s.train.label); yv = le.transform(s.valid.label)
    vec, model = build_bilstm(len(le.classes_)); vec.adapt(s.train.text.to_numpy())
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    model.fit(s.train.text.to_numpy(), ytr, validation_data=(s.valid.text.to_numpy(), yv), epochs=10, batch_size=64, callbacks=callbacks, verbose=2)
    vp = model.predict(s.valid.text.to_numpy(), verbose=0); tp = model.predict(s.test.text.to_numpy(), verbose=0)
    classes = le.classes_; vpred = classes[vp.argmax(1)]; tpred = classes[tp.argmax(1)]
    route = select_threshold(s.valid.label.to_numpy(), vp, classes, .90)
    test_route = evaluate_routing(s.test.label.to_numpy(), tp, classes, route.threshold)
    result = {
        "model": "tensorflow_bilstm",
        "validation": classification_metrics(s.valid.label, vpred),
        "test": classification_metrics(s.test.label, tpred),
        "selected_validation_routing": route.to_dict(),
        "test_routing": test_route.to_dict(),
    }
    model.save(out/"bilstm.keras")
    (out/"tensorflow_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", default="results")
    a = ap.parse_args(); main(a.train, a.test, a.out)
