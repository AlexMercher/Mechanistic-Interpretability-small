#!/usr/bin/env python
"""
Objective C -- cross-domain semantic relationship discovery, at the
CONCEPT (cluster) level rather than the individual-prompt level, since
individual-prompt cross-domain "nearest neighbour" picks were shown in
Objective A to be majority false positives driven by template phrasing
(see reports/embedding_space_evaluation.md). Averaging over a cluster's
members before comparing should dampen per-prompt template noise while
preserving real topical signal, but this is a HYPOTHESIS the teacher-model
review step below is specifically designed to check, not an assumption
taken on faith.

Pipeline:
  1. Compute each cluster's centroid (mean of member embeddings, re-normalized).
  2. Compute all cross-domain centroid-pair cosine similarities (595 clusters
     -> ~176k pairs, cheap).
  3. Select candidates for review: for EVERY unordered pair of domains
     (10 pairs for 5 domains), keep its top --per-domain-pair candidates by
     centroid similarity. This guarantees coverage of all domain pairs
     (not just the globally highest-similarity ones), per the task's
     requirement to investigate cross-domain relationships broadly.
  4. Write candidates (with representative prompts from both sides) for
     teacher-model review. Does NOT assign a relationship type itself --
     that is Objective C's human/teacher-reviewed step, done by a separate
     Agent review pass that writes concept_relationships.jsonl.

Usage:
    python discover_cross_domain_relations.py [--per-domain-pair 15]
"""
import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-domain-pair", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    records = sc.load_normalized_records()
    vectors, prompt_ids, _ = sc.load_embeddings()
    pid_to_row = {p: r for r, p in enumerate(prompt_ids)}

    clustering_dir = args.out_dir / "clustering"
    taxonomy_dir = args.out_dir / "taxonomy"
    taxonomy_dir.mkdir(parents=True, exist_ok=True)

    all_clusters = []
    for domain in sc.DOMAINS:
        detail = json.loads((clustering_dir / f"{domain.lower()}_cluster_details.json").read_text(encoding="utf-8"))
        for c in detail["clusters"]:
            if c["raw_record_count"] < 5:
                continue  # too small to trust a centroid
            all_clusters.append(c)

    names = json.loads((clustering_dir / "cluster_provisional_names.json").read_text(encoding="utf-8"))

    centroids = np.zeros((len(all_clusters), vectors.shape[1]), dtype=np.float32)
    for i, c in enumerate(all_clusters):
        rows = [pid_to_row[p] for p in c["prompt_ids"]]
        centroid = vectors[rows].mean(axis=0)
        centroids[i] = centroid / (np.linalg.norm(centroid) + 1e-12)

    sim_matrix = centroids @ centroids.T

    by_domain_pair = defaultdict(list)
    for i, j in combinations(range(len(all_clusters)), 2):
        ci, cj = all_clusters[i], all_clusters[j]
        if ci["domain"] == cj["domain"]:
            continue
        dom_pair = tuple(sorted([ci["domain"], cj["domain"]]))
        by_domain_pair[dom_pair].append((float(sim_matrix[i, j]), i, j))

    candidates = []
    for dom_pair, lst in sorted(by_domain_pair.items()):
        lst.sort(reverse=True)
        for sim, i, j in lst[: args.per_domain_pair]:
            ci, cj = all_clusters[i], all_clusters[j]
            candidates.append({
                "candidate_id": f"{ci['cluster_id']}__{cj['cluster_id']}",
                "domain_pair": list(dom_pair),
                "centroid_cosine_similarity": sim,
                "cluster_a": ci["cluster_id"], "cluster_b": cj["cluster_id"],
                "subdomain_a": ci["subdomain"], "subdomain_b": cj["subdomain"],
                "provisional_name_a": names[ci["cluster_id"]]["provisional_name"],
                "provisional_name_b": names[cj["cluster_id"]]["provisional_name"],
                "raw_record_count_a": ci["raw_record_count"], "raw_record_count_b": cj["raw_record_count"],
                "representative_prompts_a": [records[p]["prompt"] for p in ci["representative_prompt_ids"][:3]],
                "representative_prompts_b": [records[p]["prompt"] for p in cj["representative_prompt_ids"][:3]],
            })

    out_path = taxonomy_dir / "cross_domain_candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[discover_cross_domain_relations] {len(all_clusters)} clusters (>=5 records) considered")
    print(f"[discover_cross_domain_relations] wrote {len(candidates)} candidate cross-domain pairs "
          f"across {len(by_domain_pair)} domain pairs to {out_path}")
    for dom_pair in sorted(by_domain_pair):
        top_sim = by_domain_pair[dom_pair][0][0] if by_domain_pair[dom_pair] else None
        print(f"  {dom_pair}: {len(by_domain_pair[dom_pair])} total pairs available, "
              f"top centroid similarity={top_sim:.4f}" if top_sim else f"  {dom_pair}: none")


if __name__ == "__main__":
    main()
