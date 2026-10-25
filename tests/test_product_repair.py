import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline.diagnostics import loss_gradient_audit
from baseline.io import config_at
from baseline.model import binary_focal_loss, prediction_health
from baseline.training import model_directory, task_settings

ROOT = Path(__file__).resolve().parents[1]


def test_balanced_focal_corrects_1022_label_negative_gradient_dominance():
    audit = loss_gradient_audit(1022)
    assert audit["mean"]["negative_to_positive_ratio"] == pytest.approx(3063, rel=1e-5)
    assert audit["balanced"]["negative_to_positive_ratio"] == pytest.approx(1, rel=1e-5)
    assert audit["balanced"]["positive_gradient"] > 1000 * audit["mean"]["positive_gradient"]


def test_balanced_focal_handles_unseen_eval_labels_and_extreme_logits():
    logits = torch.tensor([[100., -100., 0.], [0., 0., 0.]], requires_grad=True)
    targets = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    loss = binary_focal_loss(logits, targets, alpha=.5, reduction="balanced")
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] > 0  # Negative labels still get suppressed.


def test_balanced_focal_is_invariant_to_repeated_identical_negative_outputs():
    small = binary_focal_loss(torch.tensor([[.2, -.1]]), torch.tensor([[1., 0.]]), alpha=.5, reduction="balanced")
    large = binary_focal_loss(torch.tensor([[.2] + [-.1] * 1021]),
                             torch.tensor([[1.] + [0.] * 1021]), alpha=.5, reduction="balanced")
    assert small == pytest.approx(large.item())


def test_product_repair_preserves_other_task_settings_and_uses_isolated_model(tmp_path):
    base = config_at(ROOT / "configs/baseline.json", tmp_path / "run")
    repair = config_at(ROOT / "configs/product_repair.json", tmp_path / "run")
    for task in ("hazard-category", "product-category", "hazard"):
        assert task_settings(base, task) == task_settings(repair, task)
        assert model_directory(base, task) == model_directory(repair, task)
    assert model_directory(repair, "product") != model_directory(base, "product")
    assert task_settings(repair, "product")["focal_reduction"] == "balanced"
    assert task_settings(repair, "product")["eval_strategy"] == "epoch"
    assert task_settings(repair, "product")["metric_for_best_model"] == "recall_at_5"
    assert repair["llm"] == base["llm"]  # Cache compatibility.
    assert repair["candidates"] == base["candidates"]
    invalid = copy.deepcopy(repair)
    invalid["model_dirs"]["product"] = "../outside"
    with pytest.raises(ValueError):
        model_directory(invalid, "product")


def test_health_check_detects_constant_model_and_accepts_informative_predictions():
    truth, labels = ["a", "b", "a"], ["a", "b", "c"]
    assert not prediction_health(np.array([[0, 0, 5]] * 3), truth, labels)["healthy"]
    good = prediction_health(np.array([[5, 0, 0], [0, 5, 0], [5, 0, 0]]), truth, labels)
    assert good["healthy"] and good["accuracy"] == 1 and good["distinct_predictions"] == 2
    assert not prediction_health(np.array([[np.nan, 0, 0]] * 3), truth, labels)["healthy"]


def test_balanced_focal_can_learn_a_1022_output_head():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    inputs = torch.eye(8)
    target = torch.nn.functional.one_hot(torch.arange(8), 1022).float()
    head = torch.nn.Linear(8, 1022)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.05)
    initial = binary_focal_loss(head(inputs), target, alpha=.5, reduction="balanced").item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = binary_focal_loss(head(inputs), target, alpha=.5, reduction="balanced")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(inputs)
    assert binary_focal_loss(logits, target, alpha=.5, reduction="balanced").item() < initial / 10
    labels = [str(i) for i in range(1022)]
    health = prediction_health(logits.numpy(), labels[:8], labels,
        {"threshold": .5, "top_k": 5, "max_candidates": 20})
    assert health["healthy"] and health["accuracy"] == 1 and health["candidate_recall"] == 1


def test_product_repair_smoke_training_uses_balanced_loss_and_leaves_old_model(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from baseline.data import prepare
    from baseline.io import read_json, read_jsonl, write_jsonl
    from baseline.training import train
    torch.set_num_threads(2)
    config = config_at(ROOT / "configs/product_repair.json", tmp_path / "run")
    prepare(config)
    root = Path(config["run_dir"])
    write_jsonl(root / "data/training_augmented.jsonl", read_jsonl(root / "data/training.jsonl")[:16])
    old = root / "models/product/final/sentinel.txt"
    old.parent.mkdir(parents=True)
    old.write_text("keep old model", encoding="utf-8")
    local = tmp_path / "tiny-product-bert"
    local.mkdir()
    (local / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfood\nrecall\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(local / "vocab.txt"))
    tokenizer.save_pretrained(local)
    model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=1022,
        max_position_embeddings=128))
    model.save_pretrained(local)
    config["training"].update(model_name=str(local), revision="main", max_length=64, fp16=False)
    result = train(config, "product", smoke=True)
    assert result["global_step"] == 2
    spec = read_json(root / "smoke_models/product/run.json")
    assert spec["training"]["focal_reduction"] == "balanced"
    assert spec["training"]["focal_alpha"] == .5
    assert spec["training"]["metric_for_best_model"] == "recall_at_5"
    assert old.read_text(encoding="utf-8") == "keep old model"
    assert (root / "smoke_models/product/validation_health.json").is_file()
