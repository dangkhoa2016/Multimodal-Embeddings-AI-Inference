"""Offline top-K evaluation of saved, model-consistent text/image embeddings.

Input is a JSON file with model_id, revision, config_sha256, images and queries.
Every image/query entry must repeat these three identifiers to prevent accidental
cross-model comparisons. Queries include relevant_images (ground-truth IDs).
No requests to inference services, GPU loading, or file-based image decoding.
"""

import argparse
import json
import math
from pathlib import Path


def _vector(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Embedding vector must be a non-empty list")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in value):
        raise ValueError("Embedding vector must contain only finite numbers")
    norm = math.sqrt(sum(v * v for v in value))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Embedding vector must have non-zero finite norm")
    return [v / norm for v in value]


def _identity(record, reference):
    if not isinstance(record, dict):
        raise ValueError("Embedding entry must be an object")
    for field in ("model_id", "revision", "config_sha256"):
        if record.get(field) != reference[field]:
            raise ValueError(f"Mixed or missing embedding identity: {field}")


def evaluate(dataset, *, top_k=1):
    """Return recall@K, hit@1, MRR and per-query ranking with cosine scores."""
    if not isinstance(dataset, dict):
        raise ValueError("Expected a JSON object")
    identity = {}
    for field in ("model_id", "revision", "config_sha256"):
        value = dataset.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing embedding identity: {field}")
        identity[field] = value
    images = dataset.get("images")
    queries = dataset.get("queries")
    if not isinstance(images, list) or not images:
        raise ValueError("At least one image vector is required")
    if not isinstance(queries, list) or not queries:
        raise ValueError("At least one text query is required")
    if type(top_k) is not int or not 1 <= top_k <= len(images):
        raise ValueError("top_k must be between 1 and number of images")

    index = {}
    dimension = None
    for image in images:
        _identity(image, identity)
        item_id = image.get("id")
        if not isinstance(item_id, str) or not item_id.strip() or item_id in index:
            raise ValueError("Image IDs must be unique non-empty strings")
        vector = _vector(image.get("vector"))
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("Image embedding dimension mismatch")
        index[item_id] = vector

    seen_queries = set()
    ranked_results = []
    recalls, reciprocal_ranks, hits = [], [], []
    for query in queries:
        _identity(query, identity)
        query_id = query.get("id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_queries:
            raise ValueError("Query IDs must be unique non-empty strings")
        seen_queries.add(query_id)
        vector = _vector(query.get("vector"))
        if len(vector) != dimension:
            raise ValueError("Query and image embedding dimensions differ")
        relevant = query.get("relevant_images")
        if (
            not isinstance(relevant, list) or not relevant
            or any(not isinstance(name, str) or name not in index for name in relevant)
            or len(set(relevant)) != len(relevant)
        ):
            raise ValueError("Ground-truth relevant_images must contain unique known IDs")
        correct = set(relevant)
        ranking = sorted(
            (
                (image_id, sum(a * b for a, b in zip(vector, image_vector, strict=True)))
                for image_id, image_vector in index.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        retrieved = [item_id for item_id, _ in ranking[:top_k]]
        recalls.append(len(set(retrieved) & correct) / len(correct))
        hits.append(int(ranking[0][0] in correct))
        rank = next(i for i, (name, _) in enumerate(ranking, 1) if name in correct)
        reciprocal_ranks.append(1 / rank)
        ranked_results.append({
            "query_id": query_id,
            "top_k": [{"image_id": name, "cosine": round(score, 6)}
                      for name, score in ranking[:top_k]],
            "first_relevant_rank": rank,
        })

    n = len(queries)
    return {
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "config_sha256": identity["config_sha256"],
        "image_count": len(images),
        "query_count": n,
        "top_k": top_k,
        "hit_at_1": sum(hits) / n,
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "results": ranked_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, action="append",
                        help="One saved model-specific embedding dataset; repeat for comparison")
    parser.add_argument("--require-both-models", action="store_true",
                        help="Refuse incomplete Qwen/WeMM comparisons")
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    try:
        reports = []
        workload_signatures = []
        for source in args.input:
            payload = json.loads(source.read_text(encoding="utf-8"))
            reports.append(evaluate(payload, top_k=args.top_k))
            workload_signatures.append((
                tuple(sorted(image["id"] for image in payload["images"])),
                tuple(sorted(
                    (query["id"], tuple(sorted(query["relevant_images"])))
                    for query in payload["queries"]
                )),
            ))
        models = [result["model_id"] for result in reports]
        if len(models) != len(set(models)):
            raise ValueError("Duplicate model in retrieval comparison")
        if args.require_both_models and set(models) != {
            "qwen3-vl-embedding-8b", "wemm-embedding-9b",
        }:
            raise ValueError("Both Qwen and WeMM datasets are required")
        if len(reports) > 1:
            # Comparing scores is only meaningful on matching evaluation sets.
            same_query_count = len({r["query_count"] for r in reports}) == 1
            same_image_count = len({r["image_count"] for r in reports}) == 1
            if not (same_query_count and same_image_count):
                raise ValueError("Model datasets must contain matching counts")
            if len(set(workload_signatures)) != 1:
                raise ValueError("Model datasets must contain matching labeled workloads")
        report = reports[0] if len(reports) == 1 else {
            "comparison_kind": "saved_embeddings_with_explicit_ground_truth",
            "models": reports,
        }
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Invalid retrieval dataset: {error}\n")
    # No raw vectors or input filenames in output.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
