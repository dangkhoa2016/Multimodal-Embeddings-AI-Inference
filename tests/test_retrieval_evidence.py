"""Replay saved Qwen retrieval rankings; compare model datasets offline."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_retrieval.py"


def make_model_dataset(model):
    meta = {"model_id": model, "revision": "a" * 40, "config_sha256": "b" * 64}
    return {
        **meta,
        "images": [
            {**meta, "id": "gift", "vector": [1.0, 0.0]},
            {**meta, "id": "sphere", "vector": [0.0, 1.0]},
        ],
        "queries": [
            {**meta, "id": "gift-query", "vector": [0.9, 0.1],
             "relevant_images": ["gift"]},
            {**meta, "id": "sphere-query", "vector": [0.1, 0.9],
             "relevant_images": ["sphere"]},
        ],
    }


def test_recorded_qwen_rankings_match_two_query_ground_truth():
    saved = json.loads((ROOT / "benchmarks/results/image-search.json").read_text())
    assert saved["run_kind"] == "measured_real_modal_inference"
    assert (saved["model"], saved["backend"], saved["dimension"]) == (
        "qwen3-vl-embedding-8b", "modal", 1024
    )
    expected = {
        "an orange gift box with a ribbon": "gift.webp",
        "a blue sphere": "sphere.webp",
    }
    ranking = {}
    for query in saved["queries"]:
        results = query["results"]
        assert results
        assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
        assert all(-1 <= r["cosine"] <= 1 for r in results)
        assert all(a["cosine"] >= b["cosine"] for a, b in zip(results, results[1:], strict=False))
        ranking[query["text"]] = results[0]["file"]
    assert ranking == expected
    # This is replay of a saved observation, not a new GPU measurement.


def test_comparison_requires_a_dataset_from_each_model(tmp_path):
    one = tmp_path / "qwen.json"
    one.write_text(json.dumps(make_model_dataset("qwen3-vl-embedding-8b")))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(one), "--require-both-models"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Both Qwen and WeMM" in result.stderr


def test_two_model_comparison_uses_supplied_vectors_only(tmp_path):
    paths = []
    for model in ("qwen3-vl-embedding-8b", "wemm-embedding-9b"):
        source = tmp_path / (model + ".json")
        source.write_text(json.dumps(make_model_dataset(model)))
        paths.append(source)
    response = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(paths[0]), "--input", str(paths[1]),
         "--top-k", "2", "--require-both-models"],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(response.stdout)
    assert report["comparison_kind"] == "saved_embeddings_with_explicit_ground_truth"
    assert {item["model_id"] for item in report["models"]} == {
        "qwen3-vl-embedding-8b", "wemm-embedding-9b"
    }
    assert all(item["query_count"] == 2 and item["image_count"] == 2
               and item["hit_at_1"] == 1 for item in report["models"])
    assert "vector" not in response.stdout


def test_cross_model_comparison_requires_matching_workloads(tmp_path):
    paths = []
    for model in ("qwen3-vl-embedding-8b", "wemm-embedding-9b"):
        payload = make_model_dataset(model)
        if model.startswith("wemm"):
            payload["queries"].pop()
        source = tmp_path / (model + ".json")
        source.write_text(json.dumps(payload))
        paths.append(source)
    response = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(paths[0]), "--input", str(paths[1])],
        capture_output=True, text=True,
    )
    assert response.returncode == 2
    assert "matching counts" in response.stderr

def test_same_count_but_different_ground_truth_is_rejected(tmp_path):
    one = make_model_dataset("qwen3-vl-embedding-8b")
    two = make_model_dataset("wemm-embedding-9b")
    two["queries"][0]["relevant_images"] = ["sphere"]
    inputs = []
    for name, payload in (("qwen", one), ("wemm", two)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(payload))
        inputs.append(path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(inputs[0]),
         "--input", str(inputs[1]), "--require-both-models"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "matching labeled workloads" in result.stderr
