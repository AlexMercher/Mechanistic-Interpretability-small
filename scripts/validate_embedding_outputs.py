#!/usr/bin/env python
"""
Independent validation of the full-corpus embedding output. Does not
re-run the model -- just checks that what generate_embeddings.py --mode full
produced is internally consistent and matches the normalized corpus.

Checks:
  - prompt_embeddings.npy row count == 25,000 == normalized record count.
  - every prompt_id in normalized/*.jsonl has exactly one row in embedding_index.csv.
  - embedding_row values are a permutation of range(N) (no gaps/dupes).
  - every vector is finite and has the manifest's declared dimension.
  - npy_sha256 in the manifest matches the actual file on disk.
  - status == "ok" for every row (or reports which aren't).

Usage:
    python scripts/validate_embedding_outputs.py [--out-dir experiments/duplicate_language_embedding]
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"
DEFAULT_OUT_DIR = ROOT / "experiments" / "duplicate_language_embedding"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    import numpy as np

    embeddings_dir = args.out_dir / "embeddings"
    npy_path = embeddings_dir / "prompt_embeddings.npy"
    index_path = embeddings_dir / "embedding_index.csv"
    manifest_path = embeddings_dir / "embedding_manifest.json"

    errors = []

    normalized_ids = set()
    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                normalized_ids.add(json.loads(line)["prompt_id"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors = np.load(npy_path)

    actual_sha256 = hashlib.sha256(npy_path.read_bytes()).hexdigest()
    if actual_sha256 != manifest["npy_sha256"]:
        errors.append(f"npy_sha256 mismatch: manifest={manifest['npy_sha256']} actual={actual_sha256}")

    if vectors.shape[0] != len(normalized_ids):
        errors.append(f"row count {vectors.shape[0]} != normalized record count {len(normalized_ids)}")
    if vectors.shape[1] != manifest["embedding_dimension"]:
        errors.append(f"dimension {vectors.shape[1]} != manifest dimension {manifest['embedding_dimension']}")

    index_rows = []
    with index_path.open(encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))

    index_ids = {r["prompt_id"] for r in index_rows}
    if index_ids != normalized_ids:
        missing = normalized_ids - index_ids
        extra = index_ids - normalized_ids
        errors.append(f"index prompt_id set mismatch: {len(missing)} missing, {len(extra)} extra "
                      f"(examples missing={list(missing)[:5]} extra={list(extra)[:5]})")

    row_indices = [int(r["embedding_row"]) for r in index_rows]
    if sorted(row_indices) != list(range(len(index_rows))):
        errors.append("embedding_row values are not a clean permutation of range(N) -- gaps or duplicates")

    bad_status = [r["prompt_id"] for r in index_rows if r["status"] != "ok"]
    if bad_status:
        errors.append(f"{len(bad_status)} rows have status != 'ok' (examples: {bad_status[:5]})")

    n_not_finite = int((~np.isfinite(vectors)).any(axis=1).sum())
    if n_not_finite:
        errors.append(f"{n_not_finite} vectors contain a non-finite value")

    norms = np.linalg.norm(vectors, axis=1)
    if manifest.get("normalized") and not bool(np.allclose(norms, 1.0, atol=1e-3)):
        n_bad_norm = int((~np.isclose(norms, 1.0, atol=1e-3)).sum())
        errors.append(f"{n_bad_norm} vectors are not unit-normalized despite manifest.normalized=True")

    result = {
        "total_normalized_records": len(normalized_ids),
        "embedding_rows": int(vectors.shape[0]),
        "embedding_dimension": int(vectors.shape[1]),
        "npy_sha256_verified": actual_sha256 == manifest["npy_sha256"],
        "all_rows_ok_status": len(bad_status) == 0,
        "all_vectors_finite": n_not_finite == 0,
        "all_vectors_normalized": manifest.get("normalized") and bool(np.allclose(norms, 1.0, atol=1e-3)),
        "prompt_id_mapping_complete": index_ids == normalized_ids,
        "errors": errors,
        "passed": len(errors) == 0,
    }
    print(json.dumps(result, indent=2))

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
