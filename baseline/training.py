import gc
import importlib.metadata
import platform
from pathlib import Path

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from baseline import TASKS
from baseline.candidates import candidate_report, select_candidates
from baseline.data import load_split
from baseline.io import file_hash, fingerprint, read_json, write_json, write_jsonl
from baseline.model import FocalTrainer, TextDataset, compute_metrics, prediction_health


def task_settings(config, task):
    return {**config["training"], **config.get("training_overrides", {}).get(task, {})}


def model_directory(config, task, smoke=False):
    root = Path(config["run_dir"]).resolve()
    relative = Path("smoke_models") / task if smoke else Path(config.get("model_dirs", {}).get(task, f"models/{task}"))
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Model directory must stay inside run_dir")
    return destination


def environment():
    return {"python": platform.python_version(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("torch", "transformers", "accelerate", "numpy", "scikit-learn", "nltk")},
        "cuda": torch.version.cuda, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def train(config, task, resume=False, smoke=False):
    if task not in TASKS:
        raise ValueError(task)
    set_seed(config["seed"])
    root = Path(config["run_dir"])
    settings = task_settings(config, task)
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Select the SEMEVAL_T4 Colab kernel. CPU is allowed only with --smoke.")
    train_rows = load_split(config, "training", augmented=True)
    validation_rows = load_split(config, "validation")
    labels = read_json(root / "data" / "labels.json")[task]
    if len(labels) != len(set(labels)) or any(r[task] not in set(labels) for r in train_rows):
        raise ValueError("Training labels do not match the classifier vocabulary")
    destination = model_directory(config, task, smoke)
    spec = {"training": settings, "seed": config["seed"], "labels": labels, "smoke": smoke,
            "data_sha256": file_hash(root / "data" / "training_augmented.jsonl"),
            "validation_sha256": file_hash(root / "data" / "validation.jsonl")}
    if settings.get("focal_reduction") == "balanced":
        spec["focal_implementation_version"] = 1
    spec["fingerprint"] = fingerprint(spec)
    if (destination / "run.json").exists():
        previous = read_json(destination / "run.json")
        if previous["fingerprint"] != spec["fingerprint"]:
            raise ValueError("Training settings/data changed. Choose a new run directory.")
        if (destination / "complete.json").exists():
            return {"task": task, "status": "already complete", "directory": str(destination)}
        if not resume:
            raise ValueError("Existing incomplete run. Use --resume or a new run directory.")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], revision=settings["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(settings["model_name"],
        revision=settings["revision"], num_labels=len(labels), problem_type="multi_label_classification",
        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)})
    spec["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
    spec["environment"] = environment()
    write_json(destination / "run.json", spec)
    write_json(destination / "config.json.snapshot", config)
    if smoke:
        train_rows, validation_rows = train_rows[:16], validation_rows[:8]
    strategy = "steps" if smoke else settings.get("eval_strategy", "steps")
    metric = settings.get("metric_for_best_model", "macro_f1")
    print(f"Training {task}: {len(train_rows)} samples, {len(labels)} labels; "
          f"focal reduction={settings.get('focal_reduction', 'mean')}, alpha={settings['focal_alpha']}; "
          f"best checkpoint by {metric}, evaluation={strategy}", flush=True)
    args = TrainingArguments(output_dir=str(destination / "checkpoints"),
        num_train_epochs=settings["epochs"], max_steps=2 if smoke else -1,
        per_device_train_batch_size=min(2, settings["batch_size"]) if smoke else settings["batch_size"],
        per_device_eval_batch_size=settings["eval_batch_size"],
        gradient_accumulation_steps=1 if smoke else settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
        warmup_ratio=settings["warmup_ratio"], optim="adamw_torch",
        fp16=settings["fp16"] and torch.cuda.is_available(),
        gradient_checkpointing=settings["gradient_checkpointing"],
        eval_strategy=strategy, save_strategy=strategy, eval_steps=1 if smoke else settings["save_steps"],
        save_steps=1 if smoke else settings["save_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model=metric, greater_is_better=True,
        logging_steps=1 if smoke else 25, report_to="none", seed=config["seed"], data_seed=config["seed"],
        dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available(),
        save_safetensors=True)
    length = min(64, settings["max_length"]) if smoke else settings["max_length"]
    trainer = FocalTrainer(model=model, args=args, processing_class=tokenizer,
        train_dataset=TextDataset(train_rows, tokenizer, labels, task, length),
        eval_dataset=TextDataset(validation_rows, tokenizer, labels, task, length),
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=lambda evaluation: compute_metrics(evaluation,
            true_labels=[r[task] for r in validation_rows], labels=labels),
        focal_alpha=settings["focal_alpha"], focal_gamma=settings["focal_gamma"],
        focal_reduction=settings.get("focal_reduction", "mean"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings["patience"])])
    checkpoint = get_last_checkpoint(str(destination / "checkpoints")) if resume and (destination / "checkpoints").exists() else None
    outcome = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(destination / "final"))
    tokenizer.save_pretrained(destination / "final")
    write_json(destination / "history.json", trainer.state.log_history)
    prediction = trainer.predict(trainer.eval_dataset)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    health = prediction_health(logits, [r[task] for r in validation_rows], labels, config["candidates"])
    write_json(destination / "validation_health.json", health)
    if settings.get("fail_on_collapse", False) and not smoke and not health["healthy"]:
        write_json(destination / "failed.json", {"reason": "validation_collapse", "health": health})
        raise RuntimeError(f"{task} still collapses on validation. See {destination / 'validation_health.json'}. "
                           "No complete marker was written; do not send these candidates to Gemini.")
    summary = {"task": task, "metrics": outcome.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
               "smoke": smoke, "global_step": trainer.state.global_step, "validation_health": health}
    write_json(destination / "complete.json", summary)
    if (destination / "failed.json").exists():
        (destination / "failed.json").unlink()
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def predict(config, split, limit=None):
    root = Path(config["run_dir"])
    rows = load_split(config, split)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        rows = rows[:limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prediction records deliberately exclude all gold labels.
    records = [{"id": r["id"], "title": r["title"], "text": r["text"], "candidates": {}} for r in rows]
    argmax = [{"id": r["id"]} for r in rows]
    reports, sources = {}, {}
    for task in TASKS:
        model_root = model_directory(config, task)
        settings = task_settings(config, task)
        if not (model_root / "complete.json").exists():
            raise ValueError(f"Train {task} before prediction")
        tokenizer = AutoTokenizer.from_pretrained(model_root / "final")
        model = AutoModelForSequenceClassification.from_pretrained(model_root / "final").to(device).eval()
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        sources[task] = {"run": read_json(model_root / "run.json")["fingerprint"],
                         "complete": read_json(model_root / "complete.json")}
        dataset = TextDataset(rows, tokenizer, labels, task, settings["max_length"], include_labels=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=settings["eval_batch_size"], shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                probabilities = model(**{k: v.to(device) for k, v in batch.items()}).logits.float().sigmoid().cpu().numpy()
                for values in probabilities:
                    records[offset]["candidates"][task] = select_candidates(values, labels, config["candidates"],
                        all_labels=config["candidates"]["st1_all_labels"] and task in TASKS[:2])
                    argmax[offset][task] = labels[int(values.argmax())]
                    offset += 1
        reports[task] = candidate_report(rows, records, task, labels)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tag = f"{split}.pilot-{limit}" if limit else split
    output = root / config.get("predictions_dir", "predictions") / tag
    write_jsonl(str(output) + ".candidates.jsonl", records)
    write_jsonl(str(output) + ".bert.jsonl", argmax)
    write_json(str(output) + ".candidate_report.json", reports)
    write_json(str(output) + ".sources.json", {"models": sources, "settings": config["candidates"],
                                                "data_sha256": file_hash(root / "data" / f"{split}.jsonl")})
    return {"output_prefix": str(output), "tasks": reports}
