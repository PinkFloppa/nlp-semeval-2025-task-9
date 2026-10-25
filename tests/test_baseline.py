import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baseline import TASKS
from baseline.candidates import select_candidates
from baseline.data import clean_text, combine, prepare
from baseline.eda import variant
from baseline.io import config_at, read_json, read_jsonl, write_jsonl
from baseline.llm import build_prompt, llm_predict, load_cache, parse_answer
from baseline.metrics import evaluate, task_score

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    return config_at(ROOT / "configs/baseline.json", tmp_path / "run")


def test_official_splits_and_train_only_vocabulary(config):
    report = prepare(config)
    assert report["split_counts"] == {"training": 5082, "validation": 565, "test": 997}
    assert report["tasks"]["product"]["train_labels"] == 1022
    assert report["tasks"]["product"]["unseen"]["validation"]["samples"] == 49
    root = Path(config["run_dir"]) / "data"
    train = read_jsonl(root / "training.jsonl")
    labels = read_json(root / "labels.json")
    assert labels["product"] == sorted({r["product"] for r in train})
    assert all(r["input"] == combine(r["title"], r["text"]) for r in train)
    assert clean_text(" Milk  recall\nMilk recall\n Not safe ") == "milk recall not safe"


def test_conditional_score_and_empty_hazard_subset():
    # Second product is wrong, but excluded because its hazard is also wrong.
    value = task_score(["a", "b"], ["x", "y"], ["a", "a"], ["x", "z"])
    assert value["hazard_macro_f1"] == pytest.approx(1 / 3)
    assert value["product_macro_f1_given_correct_hazard"] == 1
    assert value["score"] == pytest.approx(2 / 3)
    assert task_score(["a"], ["x"], ["b"], ["x"])["score"] == 0


def test_matches_organizers_reference_function():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(42)
    h, p, hp, pp = [rng.integers(0, 5, 100) for _ in range(4)]
    mask = hp == h
    expected = (f1_score(h, hp, average="macro") + f1_score(p[mask], pp[mask], average="macro")) / 2
    assert task_score(h, p, hp, pp)["score"] == pytest.approx(expected)


def test_candidate_threshold_fallback_cap_and_ties():
    settings = {"threshold": 0.5, "top_k": 2, "max_candidates": 2}
    labels = ["a", "b", "c"]
    assert [x["label"] for x in select_candidates([.1, .2, .3], labels, settings)] == ["c", "b"]
    assert [x["label"] for x in select_candidates([.8, .8, .9], labels, settings)] == ["c", "a"]
    assert len(select_candidates([.8, .8, .9], labels, settings, all_labels=True)) == 3
    with pytest.raises(ValueError):
        select_candidates([float("nan")], ["a"], settings)


def test_eda_protects_labels_title_negation_and_all_targets():
    row = {"title": "original title", "text": "do not eat smoked sausage with listeria monocytogenes. don't consume unsafe food.",
           "product": "smoked sausage", "hazard": "listeria monocytogenes",
           "product-category": "meat", "hazard-category": "biological"}
    before = copy.deepcopy(row)
    for operation in ("synonym_replacement", "random_insertion", "random_swap", "random_deletion"):
        text = variant(row, random.Random(42), operation, .5, synonym_fn=lambda word: ["sample"])
        assert "smoked sausage" in text and "listeria monocytogenes" in text
        assert "not" in text and "don't" in text
        assert row == before


def test_llm_prompt_parsing_cache_and_no_gold_leakage(config, tmp_path, monkeypatch):
    row = {"id": "1", "title": "recall", "text": "gold-secret is not an input field",
           "candidates": {task: [{"label": "a", "probability": .9}, {"label": "b", "probability": .1}] for task in TASKS},
           "hazard": "secret-label"}
    assert "secret-label" not in build_prompt(row, "hazard", 50000)
    options = row["candidates"]["hazard"]
    assert parse_answer('"a"', options, .9) == ("a", "exact")
    assert parse_answer("2", options, .9) == ("b", "index")
    assert parse_answer("unrelated answer", options, .9) == ("a", "bert_fallback")
    source, output = tmp_path / "candidates.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(source, [row])
    calls = []
    def fake_generate(prompt, settings, key):
        calls.append(prompt)
        return "b", {"usage": {"totalTokenCount": 5}, "model_version": "test"}
    monkeypatch.setattr("baseline.llm.generate", fake_generate)
    monkeypatch.setattr("baseline.llm.api_key", lambda: "test-key")
    monkeypatch.setattr("baseline.llm.time.sleep", lambda seconds: None)
    assert llm_predict(config, source, output, dry_run=True)["api_calls"] == 0
    assert not calls
    first = llm_predict(config, source, output)
    assert first["new_api_calls"] == 4
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", "8")
    assert llm_predict(config, source, output)["new_api_calls"] == 0
    assert len(calls) == 4
    assert read_jsonl(output)[0]["product"] == "b"


def test_cache_recovers_only_a_truncated_last_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    good = json.dumps({"cache_key": "abc", "prediction": "milk"}).encode() + b"\n"
    path.write_bytes(good + b'{"cache_key":')
    assert load_cache(path)["abc"]["prediction"] == "milk"
    assert path.read_bytes() == good
    path.write_bytes(b'bad record\n' + good)
    with pytest.raises(ValueError):
        load_cache(path)


def test_evaluation_rejects_partial_and_misaligned_predictions(config, tmp_path):
    prepare(config)
    prediction = tmp_path / "prediction.jsonl"
    write_jsonl(prediction, [{"id": "not-a-real-id"}])
    with pytest.raises(ValueError):
        evaluate(config, "validation", prediction)
    gold = read_jsonl(Path(config["run_dir"]) / "data" / "validation.jsonl")
    write_jsonl(prediction, [{"id": gold[0]["id"], **{t: gold[0][t] for t in TASKS}}])
    with pytest.raises(ValueError):
        evaluate(config, "validation", prediction)
    report = evaluate(config, "validation", prediction, allow_partial=True)
    assert report["partial"] and report["ST1"]["score"] == 1


def test_focal_loss_numerical_stability_and_gradients():
    import torch
    from baseline.model import binary_focal_loss
    logits = torch.tensor([[100., -100., 0.]], requires_grad=True)
    target = torch.tensor([[1., 0., 1.]])
    loss = binary_focal_loss(logits, target, alpha=.5, gamma=0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target) * .5
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    hard = binary_focal_loss(torch.tensor([[-5., 5.]]), torch.tensor([[1., 0.]]))
    easy = binary_focal_loss(torch.tensor([[5., -5.]]), torch.tensor([[1., 0.]]))
    assert hard > easy


def test_trainer_metrics_count_distinct_unseen_labels():
    from baseline.model import compute_metrics
    evaluation = SimpleNamespace(predictions=np.array([[3., 0.], [3., 0.], [0., 3.]]),
        label_ids=np.array([[1., 0.], [0., 0.], [0., 0.]]))
    result = compute_metrics(evaluation, true_labels=["a", "unseen-1", "unseen-2"], labels=["a", "b"])
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["recall_at_5"] == pytest.approx(1 / 3)


def test_real_tiny_bert_training_checkpoint_and_reload(config, tmp_path):
    """Offline integration: actual forward/backward, evaluation, save and reload."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from baseline.training import train
    torch.set_num_threads(2)
    prepare(config)
    data_root = Path(config["run_dir"]) / "data"
    write_jsonl(data_root / "training_augmented.jsonl", read_jsonl(data_root / "training.jsonl")[:16])
    model_path = tmp_path / "tiny-bert"
    model_path.mkdir()
    (model_path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfood\nrecall\nmilk\nsalmonella\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(model_path / "vocab.txt"))
    tokenizer.save_pretrained(model_path)
    model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=10,
        max_position_embeddings=128))
    model.save_pretrained(model_path)
    config["training"].update(model_name=str(model_path), revision="main", max_length=64, fp16=False)
    result = train(config, "hazard-category", smoke=True)
    assert result["global_step"] == 2
    destination = Path(config["run_dir"]) / "smoke_models/hazard-category"
    assert (destination / "complete.json").exists()
    assert list((destination / "checkpoints").glob("checkpoint-*/optimizer.pt"))
    loaded = BertForSequenceClassification.from_pretrained(destination / "final")
    assert loaded.config.num_labels == 10
    assert train(config, "hazard-category", smoke=True, resume=True)["status"] == "already complete"
    # Simulate interruption after the last durable checkpoint and exercise actual resume.
    (destination / "complete.json").unlink()
    assert train(config, "hazard-category", smoke=True, resume=True)["global_step"] >= 2


def test_four_classifiers_predict_and_evaluate(config, tmp_path):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from baseline.io import write_json
    from baseline.training import predict
    torch.set_num_threads(2)
    prepare(config)
    root = Path(config["run_dir"])
    labels = read_json(root / "data/labels.json")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfood\nrecall\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    for task in TASKS:
        model_root = root / "models" / task
        label_list = labels[task]
        model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
            num_labels=len(label_list), id2label=dict(enumerate(label_list)),
            label2id={label: i for i, label in enumerate(label_list)}))
        model.save_pretrained(model_root / "final")
        tokenizer.save_pretrained(model_root / "final")
        write_json(model_root / "complete.json", {"test_fixture": True})
        write_json(model_root / "run.json", {"fingerprint": "offline-test"})
    config["training"]["max_length"] = 64
    result = predict(config, "validation", limit=3)
    prefix = result["output_prefix"]
    records = read_jsonl(prefix + ".candidates.jsonl")
    assert len(records) == 3
    assert all(not set(TASKS) & set(row) for row in records)
    assert all(set(row["candidates"]) == set(TASKS) for row in records)
    report = evaluate(config, "validation", prefix + ".bert.jsonl", allow_partial=True)
    assert report["partial"] and report["samples"] == 3
    assert 0 <= report["ST2"]["score"] <= 1
