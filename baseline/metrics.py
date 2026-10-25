"""Conditional macro-F1 defined in the project PDF and task overview, Eq. 2."""
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from baseline import TASKS
from baseline.data import load_split
from baseline.io import read_jsonl, write_json


def macro_f1(truth, predictions):
    """Eq. 2: mean of per-class F1, not F1 of macro precision/recall.

    As in the organizers' sklearn reference, classes are the union of gold
    and predicted labels on the evaluated subset; undefined class F1 is zero.
    """
    return float(f1_score(truth, predictions, average="macro", zero_division=0)) if len(truth) else 0.0


def task_score(hazard, product, predicted_hazard, predicted_product):
    """Eqs. 3-4: filter both product vectors by correct hazard, then average F1."""
    if not (len(hazard) == len(product) == len(predicted_hazard) == len(predicted_product)):
        raise ValueError("Prediction lengths differ")
    mask = [i for i, (a, b) in enumerate(zip(hazard, predicted_hazard)) if a == b]
    h = macro_f1(hazard, predicted_hazard)
    p = macro_f1([product[i] for i in mask], [predicted_product[i] for i in mask])
    return {"score": (h + p) / 2, "hazard_macro_f1": h,
            "product_macro_f1_given_correct_hazard": p, "correct_hazard_samples": len(mask),
            "samples": len(hazard)}


def evaluate(config, split, predictions_path, allow_partial=False):
    gold = load_split(config, split)
    predictions = read_jsonl(predictions_path)
    by_id = {r["id"]: r for r in predictions}
    gold_ids = {r["id"] for r in gold}
    if len(by_id) != len(predictions) or set(by_id) - gold_ids:
        raise ValueError("Duplicate or unknown prediction IDs")
    if not allow_partial and set(by_id) != gold_ids:
        raise ValueError("Incomplete predictions. Use --allow-partial only for a labelled pilot report.")
    gold = [r for r in gold if r["id"] in by_id]
    if not gold:
        raise ValueError("No predictions to evaluate")
    result = {"split": split, "partial": len(gold) != len(gold_ids), "samples": len(gold), "targets": {}}
    targets = {}
    for task in TASKS:
        truth = [r[task] for r in gold]
        guessed = [by_id[r["id"]][task] for r in gold]
        if any(not isinstance(label, str) or not label for label in guessed):
            raise ValueError(f"Invalid predictions for {task}")
        targets[task] = (truth, guessed)
        result["targets"][task] = {"macro_f1": macro_f1(truth, guessed),
            "classification_report": classification_report(truth, guessed, output_dict=True, zero_division=0)}
    for name, h, p in (("ST1", "hazard-category", "product-category"), ("ST2", "hazard", "product")):
        result[name] = task_score(targets[h][0], targets[p][0], targets[h][1], targets[p][1])
    destination = Path(predictions_path).with_suffix("")
    write_json(str(destination) + ".metrics.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in TASKS[:2]:
        truth, guessed = targets[task]
        labels = sorted(set(truth) | set(guessed))
        matrix = confusion_matrix(truth, guessed, labels=labels)
        fig, ax = plt.subplots(figsize=(14, 12))
        plot = ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title=task)
        plt.setp(ax.get_xticklabels(), rotation=90)
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(str(destination) + f".{task}.png", dpi=150)
        plt.close(fig)
    return {k: v for k, v in result.items() if k != "targets"}
