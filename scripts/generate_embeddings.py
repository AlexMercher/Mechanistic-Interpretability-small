#!/usr/bin/env python
"""
Generates semantic embeddings for the normalized 25K prompt corpus using
intfloat/multilingual-e5-small (see run_metadata/model_selection.md for
model choice rationale). These are standalone semantic embeddings for
nearest-neighbour/clustering use in the next research stage -- NOT target-
model activations, and no training/SAE/clustering happens in this script.

Two modes:
  --mode pilot   stratified sample (script/language diversity, short+long,
                 all 5 domains) -- sanity-checks the pipeline before the
                 full run. Writes experiments/duplicate_language_embedding/reports/embedding_pilot_report.json.
                 Does not write to embeddings/ (that's the full-run output).
  --mode full    all 25,000 records. Resumable: records already embedded
                 under an IDENTICAL model/revision/prefix/normalization
                 config are skipped, not recomputed. A config change
                 requires --overwrite (never silently mixes configs).

Read-only against raw/normalized/processed. All outputs under
experiments/duplicate_language_embedding/.

Usage:
    python scripts/generate_embeddings.py --mode pilot
    python scripts/generate_embeddings.py --mode full
    python scripts/generate_embeddings.py --mode full --overwrite
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_utils  # noqa: E402  (must be imported before torch/transformers for cache redirection)

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"
ANNOTATIONS_PATH = ROOT / "experiments" / "duplicate_language_embedding" / "annotations" / "language_script_annotations.jsonl"
DEFAULT_OUT_DIR = ROOT / "experiments" / "duplicate_language_embedding"

PILOT_LANGUAGE_FORMS = ["native_hindi", "romanized_hindi", "hinglish", "english", "mixed_multilingual"]


def load_normalized_records():
    records = {}
    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["prompt_id"]] = rec
    return records


def load_annotations():
    annotations = {}
    with ANNOTATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            annotations[a["prompt_id"]] = a
    return annotations


def select_pilot(records: dict, annotations: dict, per_domain_form_quota: int = 2):
    """Stratified: every (domain, language_form) combination that actually occurs in the
    corpus gets up to `per_domain_form_quota` picks (shortest + longest available), so the
    pilot guarantees coverage of all 5 domains AND all observed language forms, rather than
    exhausting one domain's quota before reaching the others."""
    by_domain_form = defaultdict(list)
    for pid, rec in records.items():
        key = (rec["domain"], annotations[pid]["language_form"])
        by_domain_form[key].append(pid)

    pilot_ids = []
    for (domain, form), pids in sorted(by_domain_form.items()):
        pids_sorted = sorted(pids, key=lambda p: (len(records[p]["prompt"].split()), p))
        if per_domain_form_quota >= len(pids_sorted):
            picked = pids_sorted
        else:
            indices = sorted({0, len(pids_sorted) - 1})[:per_domain_form_quota]
            picked = [pids_sorted[i] for i in indices]
        pilot_ids.extend(picked)

    return sorted(set(pilot_ids))


def cosine_sim_matrix(vectors):
    import numpy as np
    # vectors are already L2-normalized, so dot product == cosine similarity
    return vectors @ vectors.T


def run_pilot(model, records, annotations, out_dir: Path):
    import numpy as np

    pilot_ids = select_pilot(records, annotations)
    texts = [records[pid]["prompt"] for pid in pilot_ids]
    vectors, token_counts = embed_utils.embed_texts(model, texts)

    dim_ok = vectors.shape == (len(pilot_ids), embed_utils.EMBEDDING_DIM)
    finite_ok = bool(np.isfinite(vectors).all())
    norms = np.linalg.norm(vectors, axis=1)
    norm_ok = bool(np.allclose(norms, 1.0, atol=1e-3))

    sims = cosine_sim_matrix(vectors)
    nearest_neighbors = []
    for i, pid in enumerate(pilot_ids):
        order = np.argsort(-sims[i])
        top = [j for j in order if j != i][:3]
        nearest_neighbors.append({
            "prompt_id": pid,
            "domain": records[pid]["domain"],
            "language_form": annotations[pid]["language_form"],
            "prompt_snippet": records[pid]["prompt"][:90],
            "nearest": [
                {
                    "prompt_id": pilot_ids[j],
                    "domain": records[pilot_ids[j]]["domain"],
                    "language_form": annotations[pilot_ids[j]]["language_form"],
                    "similarity": round(float(sims[i][j]), 4),
                    "prompt_snippet": records[pilot_ids[j]]["prompt"][:90],
                }
                for j in top
            ],
        })

    # cross-script comparison: closest available same-domain, different-script pair
    cross_script_pairs = []
    hindi_pilot = [pid for pid in pilot_ids if records[pid]["domain"] == "HINDI"]
    devanagari_ids = [pid for pid in hindi_pilot if annotations[pid]["language_form"] == "native_hindi"]
    romanized_ids = [pid for pid in hindi_pilot if annotations[pid]["language_form"] == "romanized_hindi"]
    if devanagari_ids and romanized_ids:
        a, b = devanagari_ids[0], romanized_ids[0]
        ia, ib = pilot_ids.index(a), pilot_ids.index(b)
        cross_script_pairs.append({
            "pair": [a, b], "forms": ["native_hindi", "romanized_hindi"],
            "similarity": round(float(sims[ia][ib]), 4),
            "note": "same HINDI domain, different script/transliteration -- not a true "
                    "parallel pair, just the closest available cross-script comparison "
                    "in the pilot sample",
        })

    # unrelated-domain sanity check: a GANDHI (history) vs SCIENCE (physics) pair should
    # NOT be highly similar
    gandhi_ids = [pid for pid in pilot_ids if records[pid]["domain"] == "GANDHI"]
    science_ids = [pid for pid in pilot_ids if records[pid]["domain"] == "SCIENCE"]
    unrelated_check = None
    if gandhi_ids and science_ids:
        a, b = gandhi_ids[0], science_ids[0]
        ia, ib = pilot_ids.index(a), pilot_ids.index(b)
        unrelated_check = {
            "pair": [a, b], "domains": ["GANDHI", "SCIENCE"],
            "similarity": round(float(sims[ia][ib]), 4),
            "expectation": "should be clearly lower than same-domain/same-topic pairs",
        }

    # similarity-compression diagnostic: sentence embedding models are known to produce
    # anisotropic spaces where even unrelated pairs score high cosine similarity. Check
    # whether that's happening here by comparing the "unrelated" cross-domain pair against
    # the overall distribution of off-diagonal pairwise similarities in the pilot.
    n = len(pilot_ids)
    off_diag = sims[~np.eye(n, dtype=bool)]
    similarity_distribution = {
        "min": round(float(off_diag.min()), 4), "mean": round(float(off_diag.mean()), 4),
        "p50": round(float(np.median(off_diag)), 4), "max": round(float(off_diag.max()), 4),
    }

    report = {
        "pilot_size": len(pilot_ids),
        "pilot_ids": pilot_ids,
        "model_name": embed_utils.MODEL_NAME,
        "model_revision": embed_utils.MODEL_REVISION,
        "embedding_dimension": embed_utils.EMBEDDING_DIM,
        "query_prefix": embed_utils.QUERY_PREFIX,
        "dimension_check_passed": dim_ok,
        "finite_check_passed": finite_ok,
        "normalization_check_passed": norm_ok,
        "language_form_counts": {
            form: sum(1 for pid in pilot_ids if annotations[pid]["language_form"] == form)
            for form in PILOT_LANGUAGE_FORMS
        },
        "domain_counts": {
            d: sum(1 for pid in pilot_ids if records[pid]["domain"] == d)
            for d in sorted({records[pid]["domain"] for pid in pilot_ids})
        },
        "token_count_range": [min(token_counts), max(token_counts)],
        "nearest_neighbor_examples": nearest_neighbors,
        "cross_script_comparison": cross_script_pairs,
        "unrelated_domain_check": unrelated_check,
        "pairwise_similarity_distribution": similarity_distribution,
        "similarity_compression_observed": (
            unrelated_check is not None
            and unrelated_check["similarity"] >= similarity_distribution["p50"] - 0.05
        ),
        "overall_pass": dim_ok and finite_ok and norm_ok,
        "caveat": "This is a technical/basic sanity check (dimensions, finiteness, "
                  "normalization, rough nearest-neighbour plausibility) -- it is NOT "
                  "proof of embedding quality, and especially not proof of equal "
                  "quality across Devanagari vs Romanized-Hindi vs English input, "
                  "per the task's explicit caution against assuming that. If "
                  "similarity_compression_observed is true, the 'unrelated' cross-domain "
                  "pair scored close to or above the median pairwise similarity in the "
                  "pilot -- a sign of the anisotropy commonly seen in sentence-embedding "
                  "models, where even unrelated texts score high raw cosine similarity. "
                  "The next milestone should treat raw cosine similarity as a RELATIVE "
                  "ranking signal (nearest-neighbour ordering), not an absolute "
                  "relatedness threshold, unless/until this is calibrated further.",
    }

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "embedding_pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[generate_embeddings:pilot] pilot_size={len(pilot_ids)} "
          f"dim_ok={dim_ok} finite_ok={finite_ok} norm_ok={norm_ok}")
    print(f"[generate_embeddings:pilot] wrote {reports_dir / 'embedding_pilot_report.json'}")
    return report["overall_pass"]


def run_full(model, records: dict, out_dir: Path, batch_size: int, overwrite: bool):
    import numpy as np
    import hashlib

    embeddings_dir = out_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    npy_path = embeddings_dir / "prompt_embeddings.npy"
    index_path = embeddings_dir / "embedding_index.csv"
    manifest_path = embeddings_dir / "embedding_manifest.json"

    config = {
        "embedding_model": embed_utils.MODEL_NAME,
        "embedding_revision": embed_utils.MODEL_REVISION,
        "embedding_dimension": embed_utils.EMBEDDING_DIM,
        "normalized": True,
        "query_prefix": embed_utils.QUERY_PREFIX,
    }

    existing_vectors = None
    existing_rows = {}  # prompt_id -> row index into existing_vectors
    if manifest_path.exists() and npy_path.exists() and index_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_config = {k: old_manifest.get(k) for k in config}
        if old_config != config:
            if not overwrite:
                print("ERROR: existing embeddings were generated with a different config:\n"
                      f"  existing: {old_config}\n  requested: {config}\n"
                      "Pass --overwrite to regenerate from scratch with the new config.", file=sys.stderr)
                sys.exit(1)
            print("[generate_embeddings:full] --overwrite given, config changed, starting fresh")
        else:
            import csv
            existing_vectors = np.load(npy_path)
            with index_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["status"] == "ok":
                        existing_rows[row["prompt_id"]] = int(row["embedding_row"])
            print(f"[generate_embeddings:full] resuming: {len(existing_rows)} prompt_ids already embedded")

    all_prompt_ids = sorted(records.keys())  # deterministic ordering
    missing_ids = [pid for pid in all_prompt_ids if pid not in existing_rows]
    print(f"[generate_embeddings:full] total={len(all_prompt_ids)} already_done={len(existing_rows)} "
          f"to_embed={len(missing_ids)}")

    new_vectors_by_id = {}
    new_token_counts = {}
    for i in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[i:i + batch_size]
        texts = [records[pid]["prompt"] for pid in batch_ids]
        vectors, token_counts = embed_utils.embed_texts(model, texts, batch_size=batch_size)
        for pid, vec, tc in zip(batch_ids, vectors, token_counts):
            new_vectors_by_id[pid] = vec
            new_token_counts[pid] = tc
        done = min(i + batch_size, len(missing_ids))
        print(f"[generate_embeddings:full] embedded {done}/{len(missing_ids)}")

    # assemble final array in deterministic prompt_id order
    final_vectors = np.zeros((len(all_prompt_ids), embed_utils.EMBEDDING_DIM), dtype=np.float32)
    rows = []
    max_seq_len = getattr(model, "max_seq_length", 512) or 512
    for row_idx, pid in enumerate(all_prompt_ids):
        if pid in existing_rows:
            vec = existing_vectors[existing_rows[pid]]
            token_count = None  # not re-measured on resume; already validated in a prior run
        else:
            vec = new_vectors_by_id[pid]
            token_count = new_token_counts[pid]
        finite_ok = bool(np.isfinite(vec).all())
        dim_ok = vec.shape[0] == embed_utils.EMBEDDING_DIM
        status = "ok" if (finite_ok and dim_ok) else "invalid"
        final_vectors[row_idx] = vec
        rows.append({
            "prompt_id": pid, "embedding_row": row_idx,
            "embedding_model": embed_utils.MODEL_NAME, "embedding_revision": embed_utils.MODEL_REVISION,
            "embedding_dimension": embed_utils.EMBEDDING_DIM, "normalized": True, "status": status,
            "token_count": token_count,
            "exceeds_max_length": (token_count is not None and token_count > max_seq_len),
        })

    np.save(npy_path, final_vectors)

    import csv
    with index_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["prompt_id", "embedding_row", "embedding_model", "embedding_revision",
                      "embedding_dimension", "normalized", "status", "token_count", "exceeds_max_length"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    npy_sha256 = hashlib.sha256(npy_path.read_bytes()).hexdigest()
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_exceeds = sum(1 for r in rows if r["exceeds_max_length"])
    manifest = {
        **config,
        "total_records": len(all_prompt_ids),
        "valid_embeddings": n_ok,
        "invalid_embeddings": len(all_prompt_ids) - n_ok,
        "records_exceeding_max_sequence_length": n_exceeds,
        "max_sequence_length": max_seq_len,
        "batch_size": batch_size,
        "device": "cpu",
        "preprocessing": "complete original `prompt` field, verbatim, with a required "
                          f"'{embed_utils.QUERY_PREFIX}' prefix prepended (e5 usage convention); "
                          "no domain/subdomain/category/annotation text appended; no truncation "
                          "performed (all prompts are well under max_sequence_length).",
        "npy_file": str(npy_path.relative_to(ROOT).as_posix()),
        "npy_sha256": npy_sha256,
        "npy_shape": list(final_vectors.shape),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[generate_embeddings:full] wrote {npy_path} shape={final_vectors.shape}")
    print(f"[generate_embeddings:full] valid_embeddings={n_ok}/{len(all_prompt_ids)} "
          f"exceeds_max_length={n_exceeds}")
    return n_ok == len(all_prompt_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true", help="full mode: discard existing embeddings if config changed")
    args = ap.parse_args()

    print("[generate_embeddings] loading normalized records ...")
    records = load_normalized_records()
    print(f"[generate_embeddings] loaded {len(records)} records")

    model = embed_utils.load_model(device=args.device)

    if args.mode == "pilot":
        annotations = load_annotations()
        ok = run_pilot(model, records, annotations, args.out_dir)
    else:
        ok = run_full(model, records, args.out_dir, args.batch_size, args.overwrite)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
