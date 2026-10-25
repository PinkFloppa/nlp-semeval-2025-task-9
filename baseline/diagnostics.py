"""Read-only evidence about a collapsed product classifier."""
from collections import Counter
from pathlib import Path

import torch

from baseline.data import load_split
from baseline.io import read_json, read_jsonl, write_json
from baseline.model import binary_focal_loss


def loss_gradient_audit(classes):
    result = {}
    for reduction, alpha in (("mean", .25), ("balanced", .5)):
        logits = torch.zeros(1, classes, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, 0] = 1
        loss = binary_focal_loss(logits, targets, alpha=alpha, reduction=reduction)
        loss.backward()
        positive = abs(logits.grad[0, 0].item())
        negative = logits.grad[0, 1:].abs().sum().item()
        result[reduction] = {"alpha": alpha, "loss_at_zero_logits": loss.item(),
            "positive_gradient": positive, "negative_gradient_mass": negative,
            "negative_to_positive_ratio": negative / positive}
    return result


def diagnose_product(config, directory=None):
    root = Path(config["run_dir"])
    predictions_dir = Path(directory) if directory else root / "predictions"
    rows = load_split(config, "validation")
    gold = {r["id"]: r for r in rows}
    classes = len(read_json(root / "data/labels.json")["product"])
    report = {"training_classes": classes, "gradient_audit": loss_gradient_audit(classes)}
    path = predictions_dir / "validation.bert.jsonl"
    if path.exists():
        predictions = read_jsonl(path)
        counts = Counter(row["product"] for row in predictions)
        if len({row["id"] for row in predictions}) != len(predictions) or any(row["id"] not in gold for row in predictions):
            raise ValueError("Invalid prediction identifiers")
        report["predictions"] = {"samples": len(predictions), "distinct_products": len(counts),
            "most_common": counts.most_common(10),
            "correct": sum(row["product"] == gold[row["id"]]["product"] for row in predictions)}
    model_root = root / "models/product"
    if (model_root / "complete.json").exists():
        report["old_training"] = read_json(model_root / "complete.json")
    if (model_root / "history.json").exists():
        report["validation_history"] = [item for item in read_json(model_root / "history.json")
                                         if "eval_macro_f1" in item]
        report["nonfinite_gradient_steps"] = [item.get("step") for item in read_json(model_root / "history.json")
            if isinstance(item.get("grad_norm"), (int, float)) and not torch.isfinite(torch.tensor(item["grad_norm"]))]
    report["interpretation"] = (
        "At zero logits the original mean focal loss has 3*(K-1) times more summed negative than positive logit-gradient magnitude. "
        "Balanced focal reduces this to 1:1. This is a controlled calculation, not a measurement of the original encoder gradients. "
        "Constant predictions plus premature checkpoint selection are established symptoms; original weights/history are needed for a complete causal audit.")
    write_json(root / "product_diagnostics.json", report)
    return report
