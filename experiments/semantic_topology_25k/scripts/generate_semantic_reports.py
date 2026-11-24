#!/usr/bin/env python
"""
Objective E -- coverage and dataset bias analysis over the discovered
concept clusters (Objective B output). Must run after build_domain_clusters.py
and name_clusters.py.

Reports, per the task's 11 required items:
  1. concepts per domain
  2. concepts per subdomain
  3. prompts per concept (raw_record_count)
  4. unique prompt texts per concept (unique_prompt_count)
  5. largest/smallest concepts
  6. unassigned prompt percentage (outliers)
  7. concepts with insufficient representation (< min_concept_size)
  8. concepts dominated by a single template family (duplicate-heavy: unique/raw ratio low)
  9. cross-domain concept relationship counts (from Objective C output, if present)
  10. language/script distribution within concepts
  11. over/underrepresented concepts (z-score on raw_record_count within domain)

Usage:
    python generate_semantic_reports.py [--min-concept-size 20]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-concept-size", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    records = sc.load_normalized_records()
    annotations = sc.load_annotations()
    clustering_dir = args.out_dir / "clustering"
    reports_dir = args.out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_clusters = []
    for domain in sc.DOMAINS:
        detail = json.loads((clustering_dir / f"{domain.lower()}_cluster_details.json").read_text(encoding="utf-8"))
        all_clusters.extend(detail["clusters"])

    pid_to_cluster = {}
    for c in all_clusters:
        for pid in c["prompt_ids"]:
            pid_to_cluster[pid] = c["cluster_id"]
    unassigned = [p for p in records if p not in pid_to_cluster]

    concepts_per_domain = Counter(c["domain"] for c in all_clusters)
    concepts_per_subdomain = Counter((c["domain"], c["subdomain"]) for c in all_clusters)

    sizes = [(c["cluster_id"], c["raw_record_count"]) for c in all_clusters]
    sizes_sorted = sorted(sizes, key=lambda x: x[1])
    smallest = sizes_sorted[:10]
    largest = sizes_sorted[-10:]

    insufficient = [c["cluster_id"] for c in all_clusters if c["raw_record_count"] < args.min_concept_size]

    template_dominated = []
    for c in all_clusters:
        if c["raw_record_count"] == 0:
            continue
        ratio = c["unique_prompt_count"] / c["raw_record_count"]
        if ratio < 0.9 and c["raw_record_count"] >= 10:
            template_dominated.append({
                "cluster_id": c["cluster_id"], "raw_record_count": c["raw_record_count"],
                "unique_prompt_count": c["unique_prompt_count"], "unique_ratio": round(ratio, 3),
            })

    # 9. cross-domain relationships (if Objective C has been run)
    rel_path = args.out_dir / "taxonomy" / "concept_relationships.jsonl"
    relationship_counts = Counter()
    relationships_per_concept = Counter()
    if rel_path.exists():
        with rel_path.open(encoding="utf-8") as f:
            for line in f:
                rel = json.loads(line)
                relationship_counts[rel["relationship_type"]] += 1
                relationships_per_concept[rel["cluster_a"]] += 1
                relationships_per_concept[rel["cluster_b"]] += 1

    # 10. language/script distribution within concepts
    lang_by_cluster = {}
    for c in all_clusters:
        langs = Counter(annotations[p]["language"] for p in c["prompt_ids"])
        scripts = Counter(annotations[p]["script"] for p in c["prompt_ids"])
        forms = Counter(annotations[p]["language_form"] for p in c["prompt_ids"])
        lang_by_cluster[c["cluster_id"]] = {
            "language": dict(langs), "script": dict(scripts), "language_form": dict(forms),
        }

    # 11. over/under-represented concepts: z-score of raw_record_count within domain
    over_under = {}
    for domain in sc.DOMAINS:
        dom_sizes = np.array([c["raw_record_count"] for c in all_clusters if c["domain"] == domain], dtype=float)
        mean, std = dom_sizes.mean(), dom_sizes.std()
        for c in all_clusters:
            if c["domain"] != domain:
                continue
            z = (c["raw_record_count"] - mean) / std if std > 0 else 0.0
            if abs(z) >= 1.5:
                over_under[c["cluster_id"]] = {"z_score": round(float(z), 2), "raw_record_count": c["raw_record_count"],
                                                "domain_mean": round(float(mean), 1)}

    coverage = {
        "n_total_concepts": len(all_clusters),
        "n_unassigned_prompts": len(unassigned),
        "unassigned_fraction": round(len(unassigned) / len(records), 4),
        "concepts_per_domain": dict(concepts_per_domain),
        "concepts_per_subdomain": {f"{d}::{s}": n for (d, s), n in concepts_per_subdomain.items()},
        "smallest_concepts": smallest,
        "largest_concepts": largest,
        "concepts_below_min_size": {"min_concept_size": args.min_concept_size, "count": len(insufficient),
                                     "cluster_ids": insufficient},
        "template_dominated_concepts": template_dominated,
        "cross_domain_relationship_counts_by_type": dict(relationship_counts),
        "relationships_per_concept_top10": relationships_per_concept.most_common(10),
        "over_under_represented_concepts": over_under,
    }

    (reports_dir / "domain_coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "cluster_metrics.json").write_text(
        json.dumps({"n_clusters": len(all_clusters),
                    "size_distribution": {"min": min(s for _, s in sizes), "max": max(s for _, s in sizes),
                                           "mean": float(np.mean([s for _, s in sizes])),
                                           "median": float(np.median([s for _, s in sizes]))}},
                   indent=2), encoding="utf-8"
    )

    lang_dist_rows = []
    for cid, dist in lang_by_cluster.items():
        lang_dist_rows.append({"cluster_id": cid, **dist})
    (reports_dir / "cross_domain_analysis.json").write_text(
        json.dumps({"language_script_distribution_per_concept": lang_dist_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("[generate_semantic_reports] wrote domain_coverage.json, cluster_metrics.json, cross_domain_analysis.json")
    print(json.dumps({k: v for k, v in coverage.items()
                      if k in ("n_total_concepts", "n_unassigned_prompts", "unassigned_fraction",
                               "concepts_per_domain")}, indent=2))
    print(f"concepts below min size ({args.min_concept_size}): {len(insufficient)}")
    print(f"template-dominated concepts (unique_ratio<0.9): {len(template_dominated)}")
    print(f"over/under-represented concepts (|z|>=1.5): {len(over_under)}")


if __name__ == "__main__":
    main()
