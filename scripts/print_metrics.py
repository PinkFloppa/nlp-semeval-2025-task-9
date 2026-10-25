"""Print full and paired metrics from downloaded baseline predictions; no API calls."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline import TASKS
from baseline.io import read_jsonl, write_json
from baseline.metrics import macro_f1, task_score


PUBLISHED_RESULTS = {
    "source": "Randl et al. (2025): SemEval-2025 Task 9: The Food Hazard Detection Challenge",
    "url": "https://aclanthology.org/2025.semeval-1.325/",
    "local_pdf": "sources/SemEval_2025_Report.pdf",
    "location": "Tables 2 and 3, PDF page 5 (proceedings page 2527)",
    "split": "test",
    "systems": [
        {"system": "TF-IDF + LR (baseline organizatorov)", "ST1": 0.498, "ST2": 0.183},
        {"system": "BERT (baseline organizatorov)", "ST1": 0.667, "ST2": 0.165},
        {"system": "Anastasia", "ST1": 0.8223, "ST2": 0.1281},
        {"system": "MyMy", "ST1": 0.8112, "ST2": 0.5278},
        {"system": "SRCB", "ST1": 0.8039, "ST2": 0.5473},
        {"system": "PATeam", "ST1": 0.8017, "ST2": 0.5266},
    ],
}


def load_gold(dataset, split):
    with Path(dataset).open(encoding="utf-8-sig", newline="") as stream:
        gold = {}
        for index, row in enumerate(csv.DictReader(stream)):
            if row["language"] == "en" and row["semeval-split"] == split:
                sample_id = row.get("") or str(index)
                if sample_id in gold:
                    raise ValueError(f"Duplicitne ID v datasete: {sample_id}")
                gold[sample_id] = row
    if not gold:
        raise ValueError(f"Dataset neobsahuje anglicke vzorky pre split {split}")
    return gold


def summarize(rows, gold):
    if not rows:
        return None
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Predikcie obsahuju duplicitne ID")
    if unknown := set(ids) - set(gold):
        raise ValueError(f"Predikcie nepatria do zvoleneho splitu. Nezname ID: {sorted(unknown)[:5]}")
    result = {"samples": len(rows), "total_samples": len(gold), "targets": {}}
    for task in TASKS:
        truth = [gold[i][task] for i in ids]
        guessed = [row[task] for row in rows]
        if any(not isinstance(x, str) or not x for x in guessed):
            raise ValueError(f"Neplatna odpoved pre {task}")
        correct = sum(a == b for a, b in zip(truth, guessed))
        counts = Counter(guessed)
        result["targets"][task] = {"correct": correct, "samples": len(rows),
            "accuracy": correct / len(rows), "macro_f1": macro_f1(truth, guessed),
            "macro_recall": float(recall_score(truth, guessed, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(truth, guessed, average="macro", zero_division=0)),
            "distinct_predictions": len(counts), "most_common_prediction": counts.most_common(1)[0][0]}
    for name, h, p in (("ST1", "hazard-category", "product-category"), ("ST2", "hazard", "product")):
        result[name] = task_score([gold[i][h] for i in ids], [gold[i][p] for i in ids],
                                  [row[h] for row in rows], [row[p] for row in rows])
    result["all_four_correct"] = sum(all(row[t] == gold[row["id"]][t] for t in TASKS) for row in rows)
    return result


def build_report(directory, dataset, split="validation"):
    directory = Path(directory)
    gold = load_gold(dataset, split)
    result = {"split": split, "total_samples": len(gold), "published_results": PUBLISHED_RESULTS}
    predictions = {}
    for model in ("bert", "gemini"):
        path = directory / f"{split}.{model}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        predictions[model] = rows
        result[model] = summarize(rows, gold)
    if not any(predictions.values()):
        raise ValueError(f"V {directory} nie su ziadne hotove {split} BERT/Gemini predikcie")
    if all(predictions.values()):
        by_id = {row["id"]: row for row in predictions["bert"]}
        common = [row for row in predictions["gemini"] if row["id"] in by_id]
        paired_bert = [by_id[row["id"]] for row in common]
        result["paired_bert"] = summarize(paired_bert, gold)
        result["paired_gemini"] = summarize(common, gold)
        result["changes"] = {}
        for task in TASKS:
            fixed = broken = 0
            for b, g in zip(paired_bert, common):
                truth = gold[g["id"]][task]
                fixed += b[task] != truth and g[task] == truth
                broken += b[task] == truth and g[task] != truth
            result["changes"][task] = {"fixed_by_gemini": fixed, "broken_by_gemini": broken}
    audit_path = directory / f"{split}.gemini.jsonl.audit.jsonl"
    if audit_path.exists():
        audit = read_jsonl(audit_path)
        result["audit"] = {"saved_responses": len(audit),
            "parse_methods": dict(Counter(row["parse_method"] for row in audit))}
    candidate_path = directory / f"{split}.candidates.jsonl"
    if candidate_path.exists():
        candidates = read_jsonl(candidate_path)
        ids = [row["id"] for row in candidates]
        if len(ids) != len(set(ids)) or set(ids) - set(gold):
            raise ValueError("Kandidati obsahuju duplicitne ID alebo ID z ineho splitu")
        if candidates:
            result["candidate_recall"] = {task: sum(any(c["label"] == gold[row["id"]][task]
                for c in row["candidates"][task]) for row in candidates) / len(candidates) for task in TASKS}
            result["candidate_samples"] = len(candidates)
    return result


def print_summary(title, summary):
    if summary is None:
        return
    print(f"\n{title}: {summary['samples']}/{summary['total_samples']} vzoriek")
    print("Doplnkove metriky jednotlivych cielov na vsetkych uvedenych vzorkach:")
    print(f"{'Ciel':<20} {'Spravne':>10} {'Accuracy':>10} {'Precision (macro)':>18} {'Recall (macro)':>15} {'Macro-F1':>10}")
    for task, values in summary["targets"].items():
        fraction = f"{values['correct']}/{values['samples']}"
        print(f"{task:<20} {fraction:>10} {values['accuracy']:>9.2%} {values['macro_precision']:>17.2%} {values['macro_recall']:>14.2%} {values['macro_f1']:>10.4f}")
    print(f"Nase namerane skore: ST1 = {summary['ST1']['score']:.4f}; ST2 = {summary['ST2']['score']:.4f}")
    print(f"Vsetky styri oznacenia spravne: {summary['all_four_correct']}/{summary['samples']}")


def print_official_scores(report):
    split_label = "TESTOVACIA MNOZINA" if report["split"] == "test" else "VALIDACNA MNOZINA"
    print(f"\nNASE NAMERANE SKORE ST1 A ST2 - {split_label}")
    print("Priemer macro-F1 hazardu a macro-F1 produktu iba pri spravnom hazarde.")
    print(f"{'Model':<42} {'Vzorky':>10} {'ST1':>10} {'ST2':>10}")
    rows = [("BERT - vsetky dostupne vzorky", report.get("bert"))]
    if report.get("paired_bert"):
        rows += [("BERT - rovnake vzorky ako Gemini", report["paired_bert"]),
                 ("BERT + Gemini - rovnake vzorky", report["paired_gemini"])]
    else:
        rows.append(("BERT + Gemini - dostupne vzorky", report.get("gemini")))
    for name, summary in rows:
        if summary:
            print(f"{name:<42} {summary['samples']:>10} {summary['ST1']['score']:>10.4f} {summary['ST2']['score']:>10.4f}")
    print()
    print(f"{'Model':<42} {'Uloha':<6} {'F1 hazard':>10} {'F1 produkt|h':>14} {'Spravny hazard':>16} {'Skore':>10}")
    for name, summary in rows:
        if not summary:
            continue
        for task in ("ST1", "ST2"):
            metric = summary[task]
            fraction = f"{metric['correct_hazard_samples']}/{metric['samples']}"
            print(f"{name:<42} {task:<6} {metric['hazard_macro_f1']:>10.4f} "
                  f"{metric['product_macro_f1_given_correct_hazard']:>14.4f} {fraction:>16} {metric['score']:>10.4f}")


def print_published_scores():
    print("\nOFICIALNE PUBLIKOVANE VYSLEDKY - TESTOVACIA MNOZINA")
    print(f"{'System z clanku':<42} {'ST1':>10} {'ST2':>10}")
    for row in PUBLISHED_RESULTS["systems"]:
        print(f"{row['system']:<42} {row['ST1']:>10.4f} {row['ST2']:>10.4f}")
    print(f"Zdroj: {PUBLISHED_RESULTS['source']}, tabulky 2 a 3.")
    print(PUBLISHED_RESULTS["url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "evaluated_predictions")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/food_recall_incidents.csv")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, help="Volitelne ulozit metriky do noveho JSON suboru")
    args = parser.parse_args()
    try:
        report = build_report(args.predictions_dir, args.dataset, args.split)
        if args.split == "validation":
            test_files = [args.predictions_dir / f"test.{model}.jsonl" for model in ("bert", "gemini")]
            report["test_report"] = build_report(args.predictions_dir, args.dataset, "test") if any(path.exists() for path in test_files) else None
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Chyba: {exc}\n")
    print(f"Split: {report['split']} | Celkom v datasete: {report['total_samples']} vzoriek")
    print("Accuracy = podiel spravnych odpovedi. ST1/ST2 pouzivaju oficialne podmienene macro-F1.")
    print_summary("Samostatny BERT - vsetky dostupne predikcie", report["bert"])
    if report.get("paired_bert"):
        print_summary("BERT - rovnake vzorky ako Gemini", report["paired_bert"])
        print_summary("BERT + Gemini - rovnake vzorky", report["paired_gemini"])
        print("\nZmeny Gemini voci BERT (opravene / pokazene):")
        for task, values in report["changes"].items():
            print(f"  {task:<20} {values['fixed_by_gemini']} / {values['broken_by_gemini']}")
    else:
        print_summary("BERT + Gemini - dostupne predikcie", report["gemini"])
    print_official_scores(report)
    if args.split == "validation":
        if report["test_report"]:
            print_official_scores(report["test_report"])
        else:
            print("\nST1/ST2 NA TESTOVACEJ MNOZINE: zatial nie su dostupne testovacie predikcie.")
            print("Pridaj test.bert.jsonl a/alebo test.gemini.jsonl. Validacne skore nie je testovacie skore.")
    print_published_scores()
    if "audit" in report:
        print(f"\nZachovane Gemini odpovede v audite: {report['audit']['saved_responses']}")
    if "candidate_recall" in report:
        print(f"\nRecall kandidatov ({report['candidate_samples']} vzoriek):")
        for task, value in report["candidate_recall"].items():
            print(f"  {task:<20} {value:.2%}")
    if report["bert"]:
        product = report["bert"]["targets"]["product"]
        if product["samples"] > 1 and product["distinct_predictions"] == 1:
            print(f"\nPROBLEM: produktovy model predikuje stale '{product['most_common_prediction']}' "
                  f"({product['samples']} vzoriek).")
    if args.output:
        sources = [args.dataset.resolve(), *args.predictions_dir.glob(f"{args.split}.*")]
        if args.output.resolve() in {path.resolve() for path in sources}:
            parser.exit(1, "Chyba: --output nesmie prepisat dataset ani zdrojove predikcie.\n")
        write_json(args.output, report)
        print(f"Metriky ulozene: {args.output}")


if __name__ == "__main__":
    main()
