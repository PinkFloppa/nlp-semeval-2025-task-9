"""Conservative EDA variants (Wei & Zou, 2019), with protected label words."""
import math
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from baseline import TASKS
from baseline.data import combine, load_split
from baseline.io import fingerprint, write_json, write_jsonl

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", re.I)
NEGATIONS = {"no", "not", "nor", "never", "without", "none", "neither", "cannot", "n't"}


@lru_cache(maxsize=20000)
def synonyms(word):
    from nltk.corpus import wordnet
    return sorted({lemma.name().lower() for syn in wordnet.synsets(word) for lemma in syn.lemmas()
                   if lemma.name().isalpha() and lemma.name().lower() != word
                   and lemma.name().lower() not in NEGATIONS})


def tokens(text):
    # Keep positions so punctuation, spacing, numbers and untouched spans are preserved.
    return list(TOKEN.finditer(text))


def variant(row, rng, operation, rate, synonym_fn=synonyms):
    text = row["text"]
    matches = tokens(text)
    protected = set(NEGATIONS)
    for task in TASKS:
        protected.update(re.findall(r"[a-z]+", row[task].lower()))
    eligible = [i for i, match in enumerate(matches) if match.group().isalpha()
                and len(match.group()) > 2 and match.group().lower() not in protected
                and not match.group().lower().endswith("n't")]
    if not eligible:
        return text
    count = min(len(eligible), max(1, math.ceil(rate * len(eligible))))
    replacements = {}
    selected = rng.sample(eligible, count)
    if operation in ("synonym_replacement", "random_insertion"):
        for i in selected:
            options = synonym_fn(matches[i].group().lower())
            if options:
                chosen = rng.choice(options)
                replacements[i] = chosen if operation == "synonym_replacement" else matches[i].group() + " " + chosen
    elif operation == "random_swap":
        for i in selected:
            # Swap only adjacent unprotected words; never cross a protected span.
            j = i + 1
            if j in eligible and i not in replacements and j not in replacements:
                replacements[i], replacements[j] = matches[j].group(), matches[i].group()
    elif operation == "random_deletion":
        if len(selected) == len(eligible):
            selected = selected[:-1]
        replacements = {i: "" for i in selected}
    else:
        raise ValueError(operation)
    for i in sorted(replacements, reverse=True):
        match = matches[i]
        text = text[:match.start()] + replacements[i] + text[match.end():]
    return re.sub(r"\s+", " ", text).strip()


def augment(config):
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError("Install WordNet first: python -m nltk.downloader wordnet omw-1.4") from exc
    settings = config["eda"]
    if not 0 < settings["change_rate"] <= 1 or settings["target_count"] < 1:
        raise ValueError("Invalid EDA rate or target count")
    rows = load_split(config, "training")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] != "training" or row["synthetic"]:
            raise ValueError("EDA accepts only original training data")
        groups[row["product"]].append(row)
    result = list(rows)
    seen = {r["input"] for r in rows}
    rng = random.Random(config["seed"])
    operations = ("synonym_replacement", "random_insertion", "random_swap", "random_deletion")
    counts = Counter()
    shortfalls = {}
    for product, originals in sorted(groups.items()):
        needed = max(0, settings["target_count"] - len(originals))
        created = 0
        for attempt in range(needed * settings["max_attempts_per_sample"]):
            if created >= needed:
                break
            original = rng.choice(originals)
            operation = operations[attempt % len(operations)]
            text = variant(original, rng, operation, settings["change_rate"])
            combined = combine(original["title"], text)
            if combined in seen:
                continue
            seen.add(combined)
            result.append({**original, "id": f"{original['id']}:eda:{len(result)}", "synthetic": True,
                           "text": text, "input": combined, "eda_operation": operation})
            counts[operation] += 1
            created += 1
        if created < needed:
            shortfalls[product] = needed - created
    root = Path(config["run_dir"]) / "data"
    write_jsonl(root / "training_augmented.jsonl", result)
    report = {"seed": config["seed"], "settings": settings, "original": len(rows),
              "synthetic": len(result) - len(rows), "total": len(result),
              "operations": dict(counts), "shortfalls": shortfalls, "fingerprint": fingerprint(result)}
    write_json(root / "augmentation.json", report)
    write_jsonl(root / "augmentation_examples.jsonl", [r for r in result if r["synthetic"]][:30])
    return report
