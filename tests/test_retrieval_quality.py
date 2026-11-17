"""Deterministic retrieval metrics and identity-isolation regression tests."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py"
SPEC = importlib.util.spec_from_file_location("evaluate_retrieval", PATH)
assert SPEC and SPEC.loader
retrieval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retrieval)


def dataset():
    meta = {
        "model_id": "qwen3-vl-embedding-8b",
        "revision": "a" * 40,
        "config_sha256": "b" * 64,
    }
    images = [
        {"id": "gift", "vector": [3.0, 0.0, 0.0]},
        {"id": "sphere", "vector": [0.0, 3.0, 0.0]},
        {"id": "triangle", "vector": [0.0, 0.0, 3.0]},
    ]
    queries = [
        {"id": "gift-search", "vector": [0.9, 0.1, 0.0],
         "relevant_images": ["gift"]},
        {"id": "sphere-search", "vector": [0.4, 0.6, 0.0],
         "relevant_images": ["sphere"]},
    ]
    return {
        **meta,
        "images": [{**meta, **item} for item in images],
        "queries": [{**meta, **item} for item in queries],
    }


def test_retrieval_top1_and_cosine_use_normalized_vectors():
    report = retrieval.evaluate(dataset(), top_k=1)
    assert report["query_count"] == 2 and report["image_count"] == 3
    assert report["hit_at_1"] == report["recall_at_k"] == report["mrr"] == 1.0
    assert [item["top_k"][0]["image_id"] for item in report["results"]] == [
        "gift", "sphere"
    ]
    assert report["results"][0]["top_k"][0]["cosine"] == pytest.approx(
        0.9 / math.sqrt(0.82), abs=1e-6
    )


def test_retrieval_rank_and_recall_k_are_distinct_statistics():
    value = dataset()
    value["queries"][0]["vector"] = [0.4, 0.6, 0.0]  # correct at rank 2
    top1 = retrieval.evaluate(value, top_k=1)
    top2 = retrieval.evaluate(value, top_k=2)
    assert top1["hit_at_1"] == 0.5
    assert top1["recall_at_k"] == 0.5
    assert top2["recall_at_k"] == 1.0
    assert top2["mrr"] == top1["mrr"] == pytest.approx(0.75)
    assert top2["results"][0]["first_relevant_rank"] == 2


def test_ties_are_deterministic_regardless_of_input_order():
    value = dataset()
    value["queries"] = [value["queries"][0]]
    value["queries"][0]["vector"] = [1.0, 1.0, 0.0]
    result1 = retrieval.evaluate(value, top_k=3)
    value["images"].reverse()
    result2 = retrieval.evaluate(value, top_k=3)
    assert result1 == result2
    assert [x["image_id"] for x in result1["results"][0]["top_k"]][:2] == [
        "gift", "sphere"
    ]


@pytest.mark.parametrize("field", ["model_id", "revision", "config_sha256"])
@pytest.mark.parametrize("kind", ["images", "queries"])
def test_cross_model_or_cross_config_vectors_are_rejected(field, kind):
    value = dataset()
    value[kind][0][field] = "different"
    with pytest.raises(ValueError, match="embedding identity"):
        retrieval.evaluate(value)


@pytest.mark.parametrize("vector", [
    [0.0, 0.0, 0.0],
    [float("nan"), 0, 1],
    [float("inf"), 0, 1],
    [True, 0, 0],
    [1, 2],
    [],
])
def test_invalid_vectors_are_not_ranked(vector):
    value = dataset()
    value["images"][0]["vector"] = vector
    with pytest.raises(ValueError, match="vector|dimension"):
        retrieval.evaluate(value)


def test_reject_duplicate_image_and_query_ids():
    value = dataset()
    value["images"][1]["id"] = value["images"][0]["id"]
    with pytest.raises(ValueError, match="Image IDs"):
        retrieval.evaluate(value)
    value = dataset()
    value["queries"][1]["id"] = value["queries"][0]["id"]
    with pytest.raises(ValueError, match="Query IDs"):
        retrieval.evaluate(value)


@pytest.mark.parametrize("relevant", [[], ["unknown"], ["gift", "gift"]])
def test_ground_truth_must_be_explicit_and_valid(relevant):
    value = dataset()
    value["queries"][0]["relevant_images"] = relevant
    with pytest.raises(ValueError, match="Ground-truth"):
        retrieval.evaluate(value)


@pytest.mark.parametrize("top_k", [0, 4, -1, 1.0, True])
def test_top_k_rejects_invalid_bounds(top_k):
    with pytest.raises(ValueError, match="top_k"):
        retrieval.evaluate(dataset(), top_k=top_k)


def test_offline_command_reports_scores_without_exposing_vectors(tmp_path, capsys):
    payload = dataset()
    source = tmp_path / "saved-embeddings.json"
    source.write_text(json.dumps(payload))
    import subprocess
    import sys
    completed = subprocess.run(
        [sys.executable, str(PATH), "--input", str(source), "--top-k", "2"],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["hit_at_1"] == 1.0
    assert result["top_k"] == 2
    assert "vector" not in completed.stdout
    assert "image_count" in completed.stdout
