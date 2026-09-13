import argparse, json
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from intent_routing.data import prepare_splits
from intent_routing.transformer_model import build_transformer
from intent_routing.evaluation import classification_metrics, per_class_report, top_confusions, expected_calibration_error
from intent_routing.routing import select_threshold, evaluate_routing

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def main(train_path, test_path, out_dir, model_name):
    np.random.seed(42); torch.manual_seed(42)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = prepare_splits(train_path, test_path)
    classes = sorted(s.train.label.unique())
    label2id = {x:i for i,x in enumerate(classes)}
    id2label = {i:x for x,i in label2id.items()}
    tokenizer, model = build_transformer(model_name, len(classes), id2label, label2id)

    def make_ds(frame):
        d = Dataset.from_dict({"text": frame.text.tolist(), "label": [label2id[x] for x in frame.label]})
        return d.map(lambda b: tokenizer(b["text"], truncation=True, max_length=64), batched=True)

    tr, va, te = make_ds(s.train), make_ds(s.valid), make_ds(s.test)

    def compute_metrics(p):
        pred = p.predictions.argmax(1)
        truth = p.label_ids
        yp = [id2label[int(i)] for i in pred]
        yt = [id2label[int(i)] for i in truth]
        return classification_metrics(yt, yp)

    args = TrainingArguments(
        output_dir=str(out/"trainer"),
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, processing_class=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    pv = trainer.predict(va); pt = trainer.predict(te)
    vp = softmax(pv.predictions); tp = softmax(pt.predictions)
    cls = np.array(classes)
    vpred = cls[vp.argmax(1)]; tpred = cls[tp.argmax(1)]
    route = select_threshold(s.valid.label.to_numpy(), vp, cls, .90)
    test_route = evaluate_routing(s.test.label.to_numpy(), tp, cls, route.threshold)
    result = {
        "model": model_name,
        "validation": classification_metrics(s.valid.label, vpred),
        "test": classification_metrics(s.test.label, tpred),
        "validation_ece": expected_calibration_error(s.valid.label.to_numpy(), vp, cls),
        "test_ece": expected_calibration_error(s.test.label.to_numpy(), tp, cls),
        "selected_validation_routing": route.to_dict(),
        "test_routing": test_route.to_dict(),
    }
    per_class_report(s.test.label, tpred).to_csv(out/"transformer_per_class.csv", index=False)
    top_confusions(s.test.label, tpred, classes).to_csv(out/"transformer_top_confusions.csv", index=False)
    trainer.save_model(str(out/"transformer_model")); tokenizer.save_pretrained(str(out/"transformer_model"))
    (out/"transformer_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", default="results"); ap.add_argument("--model-name", default="distilbert-base-uncased")
    a = ap.parse_args(); main(a.train, a.test, a.out, a.model_name)
