import pytest

from baseline import TASKS
from scripts.print_metrics import print_official_scores, summarize


def test_summary_uses_ids_instead_of_row_order_and_counts_errors():
    gold = {
        "first": {task: "a" for task in TASKS},
        "second": {task: "b" for task in TASKS},
        "third": {task: "c" for task in TASKS},
    }
    rows = [{"id": "second", **gold["second"]}, {"id": "first", **gold["first"]}]
    rows[0]["product"] = "wrong"
    result = summarize(rows, gold)
    assert result["total_samples"] == 3 and result["samples"] == 2
    assert result["targets"]["hazard"]["accuracy"] == 1
    assert result["targets"]["product"]["accuracy"] == .5
    assert result["targets"]["product"]["macro_recall"] == pytest.approx(1 / 3)
    assert result["targets"]["product"]["macro_precision"] == pytest.approx(1 / 3)
    assert result["all_four_correct"] == 1
    assert result["ST1"]["score"] == 1
    assert result["ST2"]["score"] == pytest.approx(2 / 3)


def test_summary_rejects_duplicate_unknown_ids_and_handles_empty():
    gold = {"a": {task: "label" for task in TASKS}}
    row = {"id": "a", **gold["a"]}
    assert summarize([], gold) is None
    with pytest.raises(ValueError, match="duplicitne"):
        summarize([row, row], gold)
    with pytest.raises(ValueError, match="Nezname ID"):
        summarize([{**row, "id": "unknown"}], gold)


def test_macro_recall_differs_from_accuracy_for_imbalanced_classes():
    gold = {str(i): {task: "a" if i < 3 else "b" for task in TASKS} for i in range(4)}
    rows = [{"id": str(i), **{task: "a" for task in TASKS}} for i in range(4)]
    result = summarize(rows, gold)
    assert result["targets"]["hazard"]["accuracy"] == .75
    assert result["targets"]["hazard"]["macro_recall"] == .5
    assert result["targets"]["hazard"]["macro_precision"] == .375


def test_official_breakdown_filters_product_by_its_own_hazard(capsys):
    gold = {
        "1": {"hazard-category": "a", "product-category": "x", "hazard": "h1", "product": "p1"},
        "2": {"hazard-category": "b", "product-category": "y", "hazard": "h2", "product": "p2"},
    }
    rows = [{"id": "1", **gold["1"]}, {"id": "2", **gold["2"]}]
    rows[1].update({"hazard-category": "a", "product-category": "wrong", "product": "wrong"})
    result = summarize(rows, gold)
    assert result["ST1"]["correct_hazard_samples"] == 1
    assert result["ST1"]["product_macro_f1_given_correct_hazard"] == 1
    assert result["ST1"]["score"] == pytest.approx(2 / 3)
    assert result["ST2"]["correct_hazard_samples"] == 2
    assert result["ST2"]["product_macro_f1_given_correct_hazard"] == pytest.approx(1 / 3)
    assert result["ST2"]["score"] == pytest.approx(2 / 3)
    print_official_scores({"split": "validation", "bert": result})
    output = capsys.readouterr().out
    assert "F1 produkt|h" in output and "1/2" in output and "2/2" in output


def test_macro_f1_matches_equation_2_per_class():
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    from baseline.metrics import macro_f1
    truth = ["a", "a", "a", "a", "b", "b", "c"]
    predictions = ["a", "a", "a", "b", "a", "b", "b"]
    precision, recall, _, _ = precision_recall_fscore_support(truth, predictions, zero_division=0)
    per_class = np.divide(2 * precision * recall, precision + recall,
                          out=np.zeros_like(precision), where=(precision + recall) > 0)
    assert macro_f1(truth, predictions) == pytest.approx(per_class.mean())
    harmonic_of_macro = 2 * precision.mean() * recall.mean() / (precision.mean() + recall.mean())
    assert macro_f1(truth, predictions) != pytest.approx(harmonic_of_macro)
