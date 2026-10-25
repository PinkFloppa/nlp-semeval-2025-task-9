import csv
import re
from collections import Counter
from pathlib import Path

from baseline import TASKS
from baseline.io import file_hash, read_json, read_jsonl, write_json, write_jsonl


def clean_text(text):
    # Remove duplicate whole lines, not arbitrary repeated words or substrings.
    lines = []
    seen = set()
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip().lower()
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return " ".join(lines)


def combine(title, text):
    return title + "\n\n" + text


def prepare(config):
    root = Path(config["run_dir"]) / "data"
    source_hash = file_hash(config["dataset"])
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["source_sha256"] != source_hash:
            raise ValueError("Dataset changed. Use a new run directory.")
    splits = {"training": [], "validation": [], "test": []}
    with open(config["dataset"], encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"title", "text", "language", "semeval-split", *TASKS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {required - set(reader.fieldnames or [])}")
        for index, row in enumerate(reader):
            if row["language"] != "en":
                continue
            split = row["semeval-split"]
            if split not in splits:
                raise ValueError(f"Unknown official split: {split}")
            if any(not row[task].strip() for task in TASKS):
                raise ValueError(f"Missing target at row {index}")
            title, text = clean_text(row["title"]), clean_text(row["text"])
            sample_id = row.get("") or str(index)
            splits[split].append({"id": sample_id, "parent_id": sample_id,
                "synthetic": False, "split": split, "title": title, "text": text,
                "input": combine(title, text), **{task: row[task] for task in TASKS}})
    if any(not rows for rows in splits.values()):
        raise ValueError("All three official splits must be nonempty.")
    ids = [r["id"] for rows in splits.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample identifiers.")
    labels = {task: sorted({r[task] for r in splits["training"]}) for task in TASKS}
    report = {"source_sha256": source_hash, "language": "en",
              "split_counts": {key: len(rows) for key, rows in splits.items()}, "tasks": {}}
    for task in TASKS:
        counts = Counter(r[task] for r in splits["training"])
        report["tasks"][task] = {"train_labels": len(counts), "train_singletons": sum(n == 1 for n in counts.values()),
            "unseen": {split: {"labels": sorted({r[task] for r in rows} - set(labels[task])),
                                "samples": sum(r[task] not in labels[task] for r in rows)}
                       for split, rows in splits.items() if split != "training"}}
    # Audit identical cleaned texts across splits without altering official splits.
    train_inputs = {r["input"] for r in splits["training"]}
    report["identical_train_texts"] = {s: sum(r["input"] in train_inputs for r in splits[s])
                                       for s in ("validation", "test")}
    for split, rows in splits.items():
        write_jsonl(root / f"{split}.jsonl", rows)
    write_json(root / "labels.json", labels)
    write_json(root / "manifest.json", report)
    return report


def load_split(config, split, augmented=False):
    name = "training_augmented" if augmented and split == "training" else split
    return read_jsonl(Path(config["run_dir"]) / "data" / f"{name}.jsonl")
