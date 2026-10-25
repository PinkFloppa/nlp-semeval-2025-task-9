import numpy as np


def select_candidates(probabilities, labels, settings, all_labels=False):
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(labels),) or not len(labels) or not np.isfinite(values).all():
        raise ValueError("Invalid probability vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Scores must be sigmoid probabilities")
    if not 0 <= settings["threshold"] <= 1 or min(settings["top_k"], settings["max_candidates"]) < 1:
        raise ValueError("Invalid candidate settings")
    order = np.argsort(-values, kind="stable")
    if all_labels:
        selected = order
    else:
        selected = [i for i in order if values[i] > settings["threshold"]]
        if not len(selected):
            selected = order[:settings["top_k"]]
        selected = selected[:settings["max_candidates"]]
    return [{"label": labels[int(i)], "probability": float(values[i])} for i in selected]


def candidate_report(rows, records, task, vocabulary):
    by_id = {r["id"]: r for r in records}
    seen = set(vocabulary)
    hits = sum(any(c["label"] == r[task] for c in by_id[r["id"]]["candidates"][task]) for r in rows)
    known = sum(r[task] in seen for r in rows)
    return {"samples": len(rows), "recall": hits / len(rows),
            "recall_seen_labels": hits / known if known else 0.0, "unseen_samples": len(rows) - known,
            "mean_candidates": float(np.mean([len(r["candidates"][task]) for r in records]))}
