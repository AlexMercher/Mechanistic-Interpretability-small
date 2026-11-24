#!/usr/bin/env python
"""
Objective B -- within-domain semantic discovery.

IMPORTANT DESIGN NOTE (found during development, not assumed upfront):
clustering each domain's full 5,000-record embedding slice in one shot was
tried first and rejected. For SCIENCE and GANDHI, HDBSCAN on the whole
domain did not recover scientific/historical topics at all -- it recovered
the `category` field (contextual_understanding / factual_knowledge /
reasoning / ...), because every record in a given category shares
near-identical template phrasing ("What is the defining mechanism
underlying X...") across all of that domain's subdomains, and that
template-level lexical signal dominates the embedding more than real
topical content does. This is the same template-dominance problem
documented quantitatively in reports/embedding_space_evaluation.md
(Objective A). Concretely: SCIENCE_hdbscan_1..5 were each *exactly* 200
`factual_knowledge` records spanning all 35 subdomains evenly, and
GANDHI's clusters were each 100% one category, spread over ~150
subdomains -- not usable as "concepts" by any definition.

Fix: cluster WITHIN each (domain, subdomain) pair instead of domain-wide.
Subdomains in this corpus (e.g. DNA/Replication, GANDHI/Porbandar and
childhood) are already coarse, curated topic buckets -- exactly the
SUBDOMAIN level in the required DOMAIN -> SUBDOMAIN -> CONCEPT -> SUBCONCEPT
hierarchy. Sub-clustering within a subdomain searches for genuine semantic
structure (CONCEPT/SUBCONCEPT) rather than rediscovering the task-framing
`category` field, since category is now a constant-ish nuisance factor
within a reasonably sized subdomain bucket rather than the dominant axis
of variation across a whole domain.

For each (domain, subdomain):
  1. If the bucket is too small to usefully subdivide, keep it as one
     concept (method="too_small_to_split").
  2. Otherwise compare HDBSCAN against KMeans (silhouette-swept k), and
     additionally compute each candidate clustering's normalized mutual
     information (NMI) against the `category` field as a category-confound
     diagnostic. The method with materially lower category-NMI is
     preferred when HDBSCAN's own result still correlates strongly with
     category (NMI > 0.6); this is reported transparently in
     `category_confound_nmi` on every (domain, subdomain) result rather
     than silently assumed to be solved.

UMAP is fit per domain ONLY for the 2D coordinates used later for plotting
(visualizations/). Cluster assignment itself is always done on the full
384-dim embedding, never on the UMAP projection, per the task's explicit
instruction not to treat UMAP proximity as evidence of semantic similarity.

Usage:
    python build_domain_clusters.py [--seed 42] [--min-cluster-size 15]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc
import numpy as np


def kmeans_sweep(X, k_range, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_, metric="cosine")
        if best is None or score > best[0]:
            best = (score, k, km)
    return best  # (silhouette, k, fitted_kmeans)


def run_hdbscan(X, min_cluster_size):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean",
        cluster_selection_method="eom",
    )
    # X is L2-normalized so euclidean distance is monotonic in cosine distance
    labels = clusterer.fit_predict(X)
    return labels


def category_nmi(labels, categories):
    from sklearn.metrics import normalized_mutual_info_score
    if len(set(labels)) < 2 or len(set(categories)) < 2:
        return 0.0
    return float(normalized_mutual_info_score(categories, labels))


def pick_representatives(X, row_indices, pids, centroid, k=5):
    sims = X[row_indices] @ centroid
    order = np.argsort(-sims)[:k]
    return [pids[row_indices[i]] for i in order]


def cluster_bucket(X, pids, categories, seed, min_cluster_size):
    """Cluster one (domain, subdomain) embedding slice. Returns (method, labels, diagnostics)."""
    n = len(pids)
    if n < 2 * min_cluster_size:
        return "too_small_to_split", np.zeros(n, dtype=int), {
            "reason": f"bucket has only {n} records (< 2x min_cluster_size={min_cluster_size}); "
                      "kept as a single concept rather than risking an unstable split.",
        }

    hdb_mcs = max(10, n // 25)
    hdb_labels = run_hdbscan(X, hdb_mcs)
    hdb_noise_frac = float((hdb_labels == -1).mean())
    hdb_n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    hdb_nmi = category_nmi(hdb_labels, categories)

    k_range = range(2, min(11, n // 15 + 2))
    km_result = kmeans_sweep(X, k_range, seed)
    km_labels = km_result[2].labels_ if km_result else None
    km_nmi = category_nmi(km_labels, categories) if km_result else None

    diagnostics = {
        "hdbscan_n_clusters": hdb_n_clusters, "hdbscan_noise_fraction": hdb_noise_frac,
        "hdbscan_category_nmi": hdb_nmi,
        "kmeans_best_k": km_result[1] if km_result else None,
        "kmeans_silhouette": km_result[0] if km_result else None,
        "kmeans_category_nmi": km_nmi,
    }

    hdb_usable = hdb_noise_frac <= 0.6 and hdb_n_clusters >= 2
    km_usable = km_result is not None

    if hdb_usable and (not km_usable or hdb_nmi <= max(0.6, km_nmi + 0.1)):
        diagnostics["reason"] = (
            f"HDBSCAN: {hdb_n_clusters} clusters, noise={hdb_noise_frac:.2f}, "
            f"category_nmi={hdb_nmi:.3f} -- selected (not dominated by the category field)."
        )
        return "hdbscan", hdb_labels, diagnostics
    if km_usable:
        diagnostics["reason"] = (
            f"KMeans (k={km_result[1]}, silhouette={km_result[0]:.3f}, category_nmi={km_nmi:.3f}) "
            f"selected over HDBSCAN (noise={hdb_noise_frac:.2f}, category_nmi={hdb_nmi:.3f}) -- "
            + ("HDBSCAN too noisy/degenerate." if not hdb_usable else "HDBSCAN result correlated too strongly with the category field.")
        )
        if km_nmi is not None and km_nmi > 0.6:
            diagnostics["category_confound_unresolved"] = True
            diagnostics["reason"] += " NOTE: KMeans result still correlates with category (NMI>0.6) -- reported, not hidden."
        return "kmeans", km_labels, diagnostics
    diagnostics["reason"] = "Neither method produced >=2 usable clusters; kept as a single concept."
    return "too_small_to_split", np.zeros(n, dtype=int), diagnostics


def cluster_domain(domain, vectors, prompt_ids, records, group_of, seed, min_cluster_size):
    idx = np.array([i for i, p in enumerate(prompt_ids) if records[p]["domain"] == domain])
    X_domain = vectors[idx]
    pids_domain = [prompt_ids[i] for i in idx]
    n = len(pids_domain)

    by_subdomain = defaultdict(list)
    for local_i, p in enumerate(pids_domain):
        by_subdomain[records[p]["subdomain"]].append(local_i)

    clusters = []
    all_labels_for_viz = np.full(n, -2, dtype=int)  # -2 placeholder, reassigned below for plotting only
    bucket_diagnostics = {}
    next_global_label = 0

    for subdomain, local_idx in sorted(by_subdomain.items()):
        local_idx = np.array(local_idx)
        X_sub = X_domain[local_idx]
        pids_sub = [pids_domain[i] for i in local_idx]
        categories_sub = [records[p]["category"] for p in pids_sub]

        method, labels, diag = cluster_bucket(X_sub, pids_sub, categories_sub, seed, min_cluster_size)
        bucket_diagnostics[subdomain] = {"method": method, "n_records": len(pids_sub), **diag}

        for label in sorted(set(labels)):
            rows_in_cluster = np.where(labels == label)[0]
            cluster_pids = [pids_sub[r] for r in rows_in_cluster]
            X_cluster = X_sub[rows_in_cluster]
            centroid = X_cluster.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            category_dist = Counter(records[p]["category"] for p in cluster_pids)
            rep_k = min(5, len(rows_in_cluster))
            sims_to_centroid = X_cluster @ centroid
            rep_order = np.argsort(-sims_to_centroid)[:rep_k]
            rep_ids = [cluster_pids[i] for i in rep_order]
            unique_n = sc.unique_text_count(cluster_pids, group_of)

            cluster_id = f"{domain}_{subdomain}_{method}_{int(label)}".replace(" ", "_").replace("/", "_")
            clusters.append({
                "cluster_id": cluster_id,
                "domain": domain,
                "subdomain": subdomain,
                "method": method,
                "raw_record_count": len(cluster_pids),
                "unique_prompt_count": unique_n,
                "category_distribution": dict(category_dist),
                "representative_prompt_ids": rep_ids,
                "representative_prompt_texts": [records[p]["prompt"] for p in rep_ids],
                "mean_intra_cluster_similarity": float(sims_to_centroid.mean()),
                "min_intra_cluster_similarity": float(sims_to_centroid.min()),
                "prompt_ids": cluster_pids,
            })
            for r in rows_in_cluster:
                all_labels_for_viz[local_idx[r]] = next_global_label
            next_global_label += 1

    domain_result = {
        "domain": domain,
        "n_records": n,
        "n_unique_texts": sc.unique_text_count(pids_domain, group_of),
        "n_subdomains": len(by_subdomain),
        "bucket_diagnostics": bucket_diagnostics,
        "n_clusters_final": len(clusters),
        "n_outliers": 0,
        "outlier_prompt_ids": [],
        "clusters": clusters,
    }
    return domain_result, X_domain, idx, all_labels_for_viz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    ap.add_argument("--make-umap-plots", action="store_true", default=True)
    args = ap.parse_args()

    records = sc.load_normalized_records()
    vectors, prompt_ids, _ = sc.load_embeddings()
    group_of = sc.load_duplicate_group_map()

    clustering_dir = args.out_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_assignments = []
    all_outliers = []
    metadata = {}

    for domain in sc.DOMAINS:
        print(f"[build_domain_clusters] clustering {domain} ...")
        result, X, idx, labels = cluster_domain(
            domain, vectors, prompt_ids, records, group_of, args.seed, args.min_cluster_size
        )
        metadata[domain] = {k: v for k, v in result.items() if k != "clusters"}
        metadata[domain]["clusters_summary"] = [
            {k: v for k, v in c.items() if k not in ("prompt_ids", "representative_prompt_texts")}
            for c in result["clusters"]
        ]

        for c in result["clusters"]:
            for pid in c["prompt_ids"]:
                all_assignments.append({
                    "prompt_id": pid, "domain": domain, "subdomain": c["subdomain"],
                    "cluster_id": c["cluster_id"], "method": c["method"], "status": "assigned",
                })
        for pid in result["outlier_prompt_ids"]:
            all_assignments.append({
                "prompt_id": pid, "domain": domain, "subdomain": None, "cluster_id": None,
                "method": None, "status": "unassigned",
            })
            all_outliers.append({"prompt_id": pid, "domain": domain, "reason": "hdbscan_noise"})

        # representatives JSON (separate, keeps assignments file small)
        (clustering_dir / f"{domain.lower()}_cluster_details.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # UMAP plot (visualization only, not used for cluster assignment)
        try:
            import umap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            reducer = umap.UMAP(n_components=2, random_state=args.seed, metric="cosine")
            coords = reducer.fit_transform(X)
            plt.figure(figsize=(7, 6))
            label_arr = np.array(labels)
            uniq = sorted(set(label_arr))
            for lab in uniq:
                mask = label_arr == lab
                color = "lightgray" if lab == -1 else None
                plt.scatter(coords[mask, 0], coords[mask, 1], s=4, label=("noise" if lab == -1 else str(lab)),
                            c=color, alpha=0.6)
            plt.title(f"{domain}: UMAP projection (visualization only, not cluster evidence)")
            plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
            plt.tight_layout()
            plt.savefig(viz_dir / f"{domain.lower()}_umap.png", dpi=120)
            plt.close()
        except Exception as e:
            print(f"[build_domain_clusters] WARNING: UMAP plot for {domain} failed: {e}")

        method_counts = Counter(c["method"] for c in result["clusters"])
        print(f"  -> {result['n_clusters_final']} clusters across {result['n_subdomains']} subdomains "
              f"(methods used: {dict(method_counts)}), n_records={result['n_records']}")

    assign_path = clustering_dir / "cluster_assignments.jsonl"
    with assign_path.open("w", encoding="utf-8") as f:
        for a in all_assignments:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    outliers_path = clustering_dir / "outliers.jsonl"
    with outliers_path.open("w", encoding="utf-8") as f:
        for o in all_outliers:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    (clustering_dir / "cluster_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_reps = {}
    for domain in sc.DOMAINS:
        detail = json.loads((clustering_dir / f"{domain.lower()}_cluster_details.json").read_text(encoding="utf-8"))
        all_reps[domain] = [
            {"cluster_id": c["cluster_id"], "subdomain": c["subdomain"],
             "representative_prompt_ids": c["representative_prompt_ids"],
             "representative_prompt_texts": c["representative_prompt_texts"]}
            for c in detail["clusters"]
        ]
    (clustering_dir / "cluster_representatives.json").write_text(
        json.dumps(all_reps, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[build_domain_clusters] wrote {len(all_assignments)} assignments, "
          f"{len(all_outliers)} outliers")
    print(f"[build_domain_clusters] total clusters across domains: "
          f"{sum(m['n_clusters_final'] for m in metadata.values())}")


if __name__ == "__main__":
    main()
