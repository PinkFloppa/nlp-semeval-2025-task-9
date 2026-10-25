import argparse
import json
from pathlib import Path

from baseline import TASKS
from baseline.io import config_at, write_json


def main():
    parser = argparse.ArgumentParser(description="SemEval-2025 Task 9 baseline")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--run-dir", help="Override artifact directory (e.g. Google Drive)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("augment")
    sub.add_parser("gemini-models")
    diagnosis = sub.add_parser("diagnose-product")
    diagnosis.add_argument("--predictions-dir", help="Optional downloaded prediction directory")
    training = sub.add_parser("train")
    training.add_argument("--task", choices=[*TASKS, "all"], default="all")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--smoke", action="store_true", help="Only two steps; outputs never used as baseline models")
    prediction = sub.add_parser("predict")
    prediction.add_argument("--split", choices=["validation", "test"], default="validation")
    prediction.add_argument("--limit", type=int)
    llm = sub.add_parser("llm")
    llm.add_argument("--candidates", required=True)
    llm.add_argument("--output", required=True)
    llm.add_argument("--limit", type=int)
    llm.add_argument("--dry-run", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = config_at(args.config, args.run_dir)
    if args.command == "prepare":
        from baseline.data import prepare
        result = prepare(config)
        write_json(Path(config["run_dir"]) / "config.snapshot.json", config)
    elif args.command == "augment":
        from baseline.eda import augment
        result = augment(config)
    elif args.command == "train":
        from baseline.training import train
        result = [train(config, task, args.resume, args.smoke) for task in (TASKS if args.task == "all" else [args.task])]
    elif args.command == "predict":
        from baseline.training import predict
        result = predict(config, args.split, args.limit)
    elif args.command == "llm":
        from baseline.llm import llm_predict
        result = llm_predict(config, args.candidates, args.output, args.limit, args.dry_run)
    elif args.command == "gemini-models":
        from baseline.llm import models
        result = models()
    elif args.command == "diagnose-product":
        from baseline.diagnostics import diagnose_product
        result = diagnose_product(config, args.predictions_dir)
    else:
        from baseline.metrics import evaluate
        result = evaluate(config, args.split, args.predictions, args.allow_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
