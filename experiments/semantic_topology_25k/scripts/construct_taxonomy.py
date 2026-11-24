#!/usr/bin/env python
"""
Objective D -- construct the final semantic hierarchy from Objective B
(within-domain clusters) and Objective C (cross-domain relationships).

Produces TWO complementary structures, per the task's explicit instruction
not to force a strict tree:

1. Domain-specific taxonomy: DOMAIN -> SUBDOMAIN -> CONCEPT, where each
   CONCEPT is one discovered cluster from build_domain_clusters.py. (A
   SUBCONCEPT level was not separately introduced: at this corpus's actual
   discovered granularity, the CONCEPT clusters themselves are already the
   finest reliably-distinguishable unit -- see run_metadata/model_selection
   notes and reports/domain_coverage.json for why a forced 5th level would
   not be evidence-backed.)
2. Cross-domain concept graph: typed relationship edges between CONCEPT
   nodes from different domains, from Objective C's teacher-reviewed
   output (taxonomy/concept_relationships.jsonl).

Every concept gets a stable concept_id independent of its (unreviewed,
auto-generated) display name. Requires build_domain_clusters.py,
name_clusters.py and discover_cross_domain_relations.py (+ its reviewed
output) to have already run.

Usage:
    python construct_taxonomy.py [--version v1]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc


def corpus_hash():
    h = hashlib.sha256()
    for jsonl_path in sorted(sc.NORMALIZED_DIR.glob("*.jsonl")):
        h.update(jsonl_path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    clustering_dir = args.out_dir / "clustering"
    taxonomy_dir = args.out_dir / "taxonomy"
    taxonomy_dir.mkdir(parents=True, exist_ok=True)

    names = json.loads((clustering_dir / "cluster_provisional_names.json").read_text(encoding="utf-8"))

    concepts = {}
    domain_tree = {d: {} for d in sc.DOMAINS}
    for domain in sc.DOMAINS:
        detail = json.loads((clustering_dir / f"{domain.lower()}_cluster_details.json").read_text(encoding="utf-8"))
        for c in detail["clusters"]:
            cid = c["cluster_id"]
            name_info = names[cid]
            concept = {
                "concept_id": cid,
                "concept_name": name_info["provisional_name"],
                "description": (
                    f"Auto-derived concept within {domain}/{c['subdomain']}, characterized by "
                    f"distinctive terms: {', '.join(name_info['distinctive_terms'])}."
                ),
                "domain": domain,
                "subdomain": c["subdomain"],
                "parent_concept_id": None,  # concepts are leaves directly under subdomain in the domain tree
                "child_concept_ids": [],
                "prompt_ids": c["prompt_ids"],
                "raw_record_count": c["raw_record_count"],
                "unique_prompt_count": c["unique_prompt_count"],
                "confidence": (
                    "low" if c["raw_record_count"] < 20 else
                    "medium" if c["raw_record_count"] < 60 else "high"
                ),
                "review_status": name_info["review_status"],
                "evidence": {
                    "representative_prompt_ids": name_info["representative_prompt_ids"],
                    "clustering_method": c["method"],
                    "mean_intra_cluster_similarity": c["mean_intra_cluster_similarity"],
                    "naming_method": name_info["naming_method"],
                },
            }
            concepts[cid] = concept
            domain_tree[domain].setdefault(c["subdomain"], []).append(cid)

    # cross-domain relationships
    relationships = []
    rel_path = taxonomy_dir / "concept_relationships.jsonl"
    if rel_path.exists():
        with rel_path.open(encoding="utf-8") as f:
            for line in f:
                relationships.append(json.loads(line))

    hierarchy = {
        "taxonomy_version": args.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_normalized_dir": "normalized/",
        "source_corpus_sha256": corpus_hash(),
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "clustering_configuration": {
            "granularity": "per (domain, subdomain) bucket, NOT domain-wide -- see "
                           "build_domain_clusters.py module docstring for why domain-wide "
                           "clustering was tried first and rejected (category-field confound)",
            "methods_considered": ["hdbscan", "kmeans (silhouette-swept k)"],
            "selection_criterion": "lower category-field NMI preferred when HDBSCAN noise>60% "
                                    "or HDBSCAN itself correlates with category (NMI>0.6)",
        },
        "review_status": (
            "Concept names are auto-generated (TF-IDF distinctive terms) and UNREVIEWED except "
            "for a spot-checked sample -- see run_metadata/ for the review sample results. "
            "Cross-domain relationships ARE individually teacher-model (Opus) reviewed with "
            "per-edge confidence and evidence; see concept_relationships.jsonl."
        ),
        "domain_specific_taxonomy": {
            domain: {
                subdomain: cluster_ids
                for subdomain, cluster_ids in subdomains.items()
            }
            for domain, subdomains in domain_tree.items()
        },
        "concepts": concepts,
        "cross_domain_relationships": relationships,
    }

    out_path = taxonomy_dir / f"semantic_hierarchy_{args.version}.json"
    out_path.write_text(json.dumps(hierarchy, indent=2, ensure_ascii=False), encoding="utf-8")

    # concept_assignments.jsonl: prompt_id -> concept_id (one row per prompt, every prompt has exactly one concept)
    assign_path = taxonomy_dir / "concept_assignments.jsonl"
    with assign_path.open("w", encoding="utf-8") as f:
        for cid, concept in concepts.items():
            for pid in concept["prompt_ids"]:
                f.write(json.dumps({
                    "prompt_id": pid, "concept_id": cid, "domain": concept["domain"],
                    "subdomain": concept["subdomain"], "confidence": concept["confidence"],
                    "assignment_type": "primary",
                }, ensure_ascii=False) + "\n")

    # concept_relationships.jsonl is already in place (written by the Objective C merge step);
    # also mirror it unchanged under taxonomy/ for completeness of the tree (no-op if same file).

    print(f"[construct_taxonomy] wrote {out_path} ({len(concepts)} concepts, {len(relationships)} cross-domain edges)")
    print(f"[construct_taxonomy] wrote {assign_path} ({sum(len(c['prompt_ids']) for c in concepts.values())} prompt assignments)")


if __name__ == "__main__":
    main()
