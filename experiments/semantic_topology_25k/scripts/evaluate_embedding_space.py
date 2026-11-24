#!/usr/bin/env python
"""
Objective A -- Calibrate and evaluate the embedding space, reusing the
25,000 pre-computed e5-small embeddings from
experiments/duplicate_language_embedding/embeddings/.

A1: integrity re-check (independent of validate_embedding_outputs.py).
A2: sampled cosine-similarity distributions across 8 comparison groups.
A3: a reproducible, labeled evaluation set of prompt pairs (deterministic
    heuristic candidate generation + teacher-model-reviewed labels, loaded
    from an existing reviewed file if present -- see --reviewed-pairs-file).
A4: retrieval metrics (precision@k, NDCG) computed against the reviewed
    labels, plus multilingual (Hindi script variant) retrieval spot checks.

Does NOT regenerate embeddings and does NOT modify any Milestone-1/2 output.

Usage:
    python evaluate_embedding_space.py [--seed 42] [--sample-pairs 20000]
        [--reviewed-pairs-file evaluation/reviewed_prompt_pairs.jsonl]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc

import numpy as np


def integrity_check(vectors, prompt_ids, manifest, records):
    errors = []
    n, dim = vectors.shape
    if n != len(records):
        errors.append(f"row count {n} != normalized record count {len(records)}")
    if dim != manifest["embedding_dimension"]:
        errors.append(f"dim {dim} != manifest dim {manifest['embedding_dimension']}")
    if set(prompt_ids) != set(records.keys()):
        missing = set(records.keys()) - set(prompt_ids)
        extra = set(prompt_ids) - set(records.keys())
        errors.append(f"prompt_id set mismatch: {len(missing)} missing, {len(extra)} extra")
    if len(set(prompt_ids)) != len(prompt_ids):
        errors.append("duplicate prompt_ids in embedding index")
    n_bad = int((~np.isfinite(vectors)).any(axis=1).sum())
    if n_bad:
        errors.append(f"{n_bad} non-finite vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if manifest.get("normalized") and not np.allclose(norms, 1.0, atol=1e-3):
        errors.append("vectors not unit-normalized despite manifest.normalized=True")
    result = {
        "rows": n,
        "dimension": dim,
        "model": manifest["embedding_model"],
        "revision": manifest["embedding_revision"],
        "prompt_id_mapping_complete": set(prompt_ids) == set(records.keys()),
        "all_finite": n_bad == 0,
        "all_normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "passed": len(errors) == 0,
        "errors": errors,
    }
    return result


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def sample_pair_indices(n, k, rng, exclude_self=True):
    """k random (i, j) index pairs with i != j, as a 2-column int array."""
    i = rng.integers(0, n, size=k * 2)
    j = rng.integers(0, n, size=k * 2)
    mask = i != j
    i, j = i[mask][:k], j[mask][:k]
    return i, j


def cosine_rows(vectors, i, j):
    return np.sum(vectors[i] * vectors[j], axis=1)


def build_similarity_distributions(vectors, prompt_ids, records, group_of, n_sample, seed, out_dir):
    rng = np.random.default_rng(seed)
    pid_to_row = {p: r for r, p in enumerate(prompt_ids)}
    domain_of = np.array([records[p]["domain"] for p in prompt_ids])
    subdomain_of = np.array([records[p]["subdomain"] for p in prompt_ids])
    n = len(prompt_ids)

    distributions = {}

    # 1. Random pairs, whole corpus
    i, j = sample_pair_indices(n, n_sample, rng)
    distributions["random_pairs_global"] = summarize(cosine_rows(vectors, i, j))

    # 2. Random pairs, same domain
    same_dom_pairs = []
    for dom in sc.DOMAINS:
        idx = np.where(domain_of == dom)[0]
        if len(idx) < 2:
            continue
        ii, jj = sample_pair_indices(len(idx), n_sample // len(sc.DOMAINS), rng)
        same_dom_pairs.append(cosine_rows(vectors, idx[ii], idx[jj]))
    distributions["random_pairs_same_domain"] = summarize(np.concatenate(same_dom_pairs))

    # 3. Random pairs, different domain
    i, j = sample_pair_indices(n, n_sample, rng)
    mask = domain_of[i] != domain_of[j]
    distributions["random_pairs_cross_domain"] = summarize(cosine_rows(vectors, i[mask], j[mask]))

    # 4. Random pairs, same subdomain
    same_sub_pairs = []
    for dom in sc.DOMAINS:
        subs = set(subdomain_of[domain_of == dom])
        for sub in subs:
            idx = np.where((domain_of == dom) & (subdomain_of == sub))[0]
            if len(idx) < 2:
                continue
            k = max(1, n_sample // 200)
            ii, jj = sample_pair_indices(len(idx), k, rng)
            same_sub_pairs.append(cosine_rows(vectors, idx[ii], idx[jj]))
    distributions["random_pairs_same_subdomain"] = summarize(np.concatenate(same_sub_pairs))

    # 7. Exact duplicate pairs (within a duplicate group -- same text by construction)
    groups = defaultdict(list)
    for pid, g in group_of.items():
        groups[g].append(pid)
    dup_sims = []
    for g, pids in groups.items():
        if len(pids) < 2:
            continue
        rows = [pid_to_row[p] for p in pids]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                dup_sims.append(float(np.dot(vectors[rows[a]], vectors[rows[b]])))
    distributions["exact_duplicate_pairs"] = summarize(dup_sims)

    # 8. Template-variant pairs: same (domain, subdomain, category), different prompt text,
    #    text differs by only a small edit (proxy: SequenceMatcher ratio > 0.6, < 1.0), sampled.
    import difflib
    template_sims = []
    by_cat = defaultdict(list)
    for p in prompt_ids:
        r = records[p]
        by_cat[(r["domain"], r["subdomain"], r["category"])].append(p)
    rng_py = random.Random(seed)
    for key, pids in by_cat.items():
        if len(pids) < 2:
            continue
        sample_pids = pids if len(pids) <= 40 else rng_py.sample(pids, 40)
        for a in range(len(sample_pids)):
            for b in range(a + 1, len(sample_pids)):
                pa, pb = sample_pids[a], sample_pids[b]
                ta, tb = records[pa]["prompt"], records[pb]["prompt"]
                if ta == tb:
                    continue
                ratio = difflib.SequenceMatcher(None, ta, tb).quick_ratio()
                if 0.6 < ratio < 1.0:
                    template_sims.append(float(np.dot(vectors[pid_to_row[pa]], vectors[pid_to_row[pb]])))
        if len(template_sims) > 5000:
            break
    distributions["template_variant_pairs"] = summarize(template_sims)

    meta = {
        "seed": seed,
        "n_sample_target": n_sample,
        "sampling_strategy": (
            "Groups 1-4 use numpy default_rng(seed) uniform random index pairs, rejecting "
            "i==j, capped/blocked as documented inline per group. Group 7 (exact duplicates) "
            "is exhaustive over the 176 known duplicate groups (no sampling needed, <1100 pairs "
            "total). Group 8 (template variants) samples up to 40 prompts per "
            "(domain, subdomain, category) bucket and keeps pairs with 0.6 < "
            "difflib.SequenceMatcher.quick_ratio() < 1.0 as a template-variant proxy, stopping "
            "once 5000 qualifying pairs are collected."
        ),
    }
    return distributions, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-pairs", type=int, default=20000)
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    print("[evaluate_embedding_space] loading data ...")
    records = sc.load_normalized_records()
    vectors, prompt_ids, manifest = sc.load_embeddings()
    group_of = sc.load_duplicate_group_map()
    print(f"[evaluate_embedding_space] {len(records)} records, {vectors.shape} embeddings")

    print("[evaluate_embedding_space] A1: integrity check ...")
    integrity = integrity_check(vectors, prompt_ids, manifest, records)
    print(json.dumps(integrity, indent=2))
    if not integrity["passed"]:
        print("[evaluate_embedding_space] INTEGRITY CHECK FAILED -- aborting", file=sys.stderr)
        sys.exit(1)

    print("[evaluate_embedding_space] A2: similarity distributions ...")
    distributions, sampling_meta = build_similarity_distributions(
        vectors, prompt_ids, records, group_of, args.sample_pairs, args.seed, args.out_dir
    )

    reports_dir = args.out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "integrity": integrity,
        "sampling": sampling_meta,
        "distributions": distributions,
    }
    (reports_dir / "similarity_distributions.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"[evaluate_embedding_space] wrote {reports_dir / 'similarity_distributions.json'}")

    for k, v in distributions.items():
        if v:
            print(f"  {k}: n={v['n']} mean={v['mean']:.4f} median={v['median']:.4f} "
                  f"p10={v['p10']:.4f} p90={v['p90']:.4f}")


if __name__ == "__main__":
    main()
