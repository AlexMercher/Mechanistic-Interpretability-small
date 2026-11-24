#!/usr/bin/env python
"""
Objective A4 -- retrieval quality evaluation, using the reviewed prompt-pair
labels from evaluation/reviewed_prompt_pairs.jsonl (produced by a teacher
model reviewing evaluation/candidate_prompt_pairs.jsonl -- see
build_evaluation_candidates.py). Cosine similarity is NOT used to produce
these labels; it is only evaluated against them here.

For every reviewed pair labeled strongly_related/related/weakly_related
with prompt_id_a as the "query": compute prompt_id_b's rank among all other
24,999 prompts by cosine similarity to prompt_id_a, then report:
  - precision@k for k in {1, 5, 10, 20}
  - NDCG@k (binary relevance, single relevant item per query)
Also separately reports on unrelated-labeled pairs: whether the embedding
model assigns them a suspiciously high raw similarity (false-neighbour
check), and on hindi_script_variant pairs specifically (multilingual
retrieval quality).

Usage:
    python evaluate_retrieval.py [--reviewed-pairs evaluation/reviewed_prompt_pairs.jsonl]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc
import numpy as np


def rank_of(vectors, pid_to_row, query_pid, target_pid):
    qrow = pid_to_row[query_pid]
    sims = vectors @ vectors[qrow]
    order = np.argsort(-sims)
    rank_positions = {row: i for i, row in enumerate(order)}
    target_row = pid_to_row[target_pid]
    # rank is 1-based, excluding the query itself (which is always rank 0 / sim 1.0)
    rank = rank_positions[target_row]
    return rank, float(sims[target_row])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reviewed-pairs", type=Path,
                     default=sc.EXP_DIR / "evaluation" / "reviewed_prompt_pairs.jsonl")
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    if not args.reviewed_pairs.exists():
        print(f"[evaluate_retrieval] {args.reviewed_pairs} not found -- "
              "run build_evaluation_candidates.py and get the pairs reviewed first", file=sys.stderr)
        sys.exit(1)

    vectors, prompt_ids, _ = sc.load_embeddings()
    pid_to_row = {p: r for r, p in enumerate(prompt_ids)}

    reviewed = [json.loads(l) for l in args.reviewed_pairs.open(encoding="utf-8")]
    print(f"[evaluate_retrieval] loaded {len(reviewed)} reviewed pairs")

    label_counts = defaultdict(int)
    cat_label_counts = defaultdict(lambda: defaultdict(int))
    for r in reviewed:
        label_counts[r["reviewed_label"]] += 1
        cat_label_counts[r["candidate_category"]][r["reviewed_label"]] += 1

    ks = [1, 5, 10, 20]
    relevant_labels = {"strongly_related", "related"}
    ranks = []
    per_pair_rows = []
    for r in reviewed:
        if r["reviewed_label"] not in relevant_labels:
            continue
        rank, sim = rank_of(vectors, pid_to_row, r["prompt_id_a"], r["prompt_id_b"])
        ranks.append(rank)
        per_pair_rows.append({**r, "rank_of_b_in_a_neighbors": rank, "cosine_similarity": sim})

    precision_at_k = {}
    ndcg_at_k = {}
    for k in ks:
        if not ranks:
            precision_at_k[k] = None
            ndcg_at_k[k] = None
            continue
        hits = [1 if rank < k else 0 for rank in ranks]
        precision_at_k[k] = sum(hits) / len(hits)
        ndcgs = [(1.0 / math.log2(rank + 2)) if rank < k else 0.0 for rank in ranks]
        ndcg_at_k[k] = sum(ndcgs) / len(ndcgs)

    # false-neighbour check: unrelated pairs with high cosine similarity
    false_neighbours = []
    for r in reviewed:
        if r["reviewed_label"] != "unrelated":
            continue
        ra, rb = pid_to_row[r["prompt_id_a"]], pid_to_row[r["prompt_id_b"]]
        sim = float(vectors[ra] @ vectors[rb])
        if sim >= 0.80:  # near/above the corpus-wide same-subdomain median -- suspicious for an "unrelated" pair
            false_neighbours.append({**r, "cosine_similarity": sim})

    # multilingual (hindi_script_variant) retrieval quality
    hsv_rows = [row for row in per_pair_rows if row["candidate_category"] == "hindi_script_variant"]
    hsv_other = []
    for r in reviewed:
        if r["candidate_category"] == "hindi_script_variant" and r["reviewed_label"] not in relevant_labels:
            ra, rb = pid_to_row[r["prompt_id_a"]], pid_to_row[r["prompt_id_b"]]
            sim = float(vectors[ra] @ vectors[rb])
            hsv_other.append({**r, "cosine_similarity": sim})

    result = {
        "n_reviewed_pairs": len(reviewed),
        "label_counts": dict(label_counts),
        "candidate_category_x_label_counts": {k: dict(v) for k, v in cat_label_counts.items()},
        "n_queries_with_relevant_partner": len(ranks),
        "precision_at_k": precision_at_k,
        "ndcg_at_k": ndcg_at_k,
        "false_neighbours_unrelated_but_high_similarity": false_neighbours,
        "hindi_script_variant_relevant_pairs": hsv_rows,
        "hindi_script_variant_other_labeled_pairs": hsv_other,
        "note": (
            "precision@k/NDCG@k use a single known relevant partner per query (the reviewed pair's "
            "other prompt) ranked against the OTHER 24,999 corpus prompts by cosine similarity; "
            "this is a minimal/partial retrieval ground truth (not exhaustive relevance judgments "
            "per query), consistent with the task's instruction to use sampling/controlled review "
            "rather than exhaustive pairwise labeling."
        ),
    }

    out_dir = args.out_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embedding_retrieval_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[evaluate_retrieval] wrote {out_dir / 'embedding_retrieval_metrics.json'}")
    print(json.dumps({k: v for k, v in result.items()
                       if k in ("label_counts", "precision_at_k", "ndcg_at_k", "n_queries_with_relevant_partner")},
                      indent=2))
    print(f"false_neighbours found: {len(false_neighbours)}")


if __name__ == "__main__":
    main()
