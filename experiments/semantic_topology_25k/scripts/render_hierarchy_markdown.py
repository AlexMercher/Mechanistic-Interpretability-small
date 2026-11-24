#!/usr/bin/env python
"""
Renders taxonomy/semantic_hierarchy_v1.json into a human-readable
taxonomy/SEMANTIC_HIERARCHY.md. Pure rendering -- no new analysis.

Usage:
    python render_hierarchy_markdown.py [--version v1]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    taxonomy_dir = args.out_dir / "taxonomy"
    hierarchy = json.loads((taxonomy_dir / f"semantic_hierarchy_{args.version}.json").read_text(encoding="utf-8"))
    concepts = hierarchy["concepts"]
    tree = hierarchy["domain_specific_taxonomy"]
    relationships = hierarchy["cross_domain_relationships"]

    lines = [
        f"# Semantic Hierarchy ({args.version})",
        "",
        f"Generated: {hierarchy['generated_at_utc']}",
        f"Source corpus sha256: `{hierarchy['source_corpus_sha256'][:16]}...`",
        f"Embedding model: `{hierarchy['embedding_model']}`@`{hierarchy['embedding_revision'][:12]}...`",
        "",
        f"**Review status:** {hierarchy['review_status']}",
        "",
        "## 1. Domain-specific taxonomy (DOMAIN -> SUBDOMAIN -> CONCEPT)",
        "",
        "Concept names are auto-generated (TF-IDF distinctive terms, see `clustering/cluster_provisional_names.json`), "
        "spot-checked by an Opus teacher-model review of a stratified 40-concept sample "
        "(`taxonomy/concept_review_results.jsonl`) -- NOT individually human-reviewed in full.",
        "",
    ]

    for domain, subdomains in tree.items():
        n_concepts = sum(len(v) for v in subdomains.values())
        lines.append(f"### {domain} ({n_concepts} concepts across {len(subdomains)} subdomains)")
        lines.append("")
        for subdomain, cids in sorted(subdomains.items()):
            lines.append(f"- **{subdomain}** ({len(cids)} concepts)")
            for cid in cids:
                c = concepts[cid]
                lines.append(
                    f"  - `{cid}` -- {c['concept_name']} "
                    f"(raw={c['raw_record_count']}, unique={c['unique_prompt_count']}, "
                    f"confidence={c['confidence']})"
                )
        lines.append("")

    lines.append("## 2. Cross-domain concept relationship graph")
    lines.append("")
    lines.append(
        f"{len(relationships)} candidate cross-domain concept pairs were teacher-model (Opus) reviewed "
        "(15 per domain-pair x 10 domain pairs, selected by centroid cosine similarity but labeled from "
        "actual prompt content, not similarity score). Breakdown by relationship type:"
    )
    lines.append("")
    by_type = defaultdict(list)
    for r in relationships:
        by_type[r["relationship_type"]].append(r)
    for rel_type, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- `{rel_type}`: {len(rows)}")
    lines.append("")
    lines.append(
        "`no_relationship` entries are retained in `concept_relationships.jsonl` as evidence of what was "
        "checked and ruled out (per-edge rationale included), not silently dropped. The graph below shows "
        "only edges with a real relationship type (excludes `no_relationship`):"
    )
    lines.append("")
    for rel_type, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if rel_type == "no_relationship":
            continue
        lines.append(f"### {rel_type} ({len(rows)})")
        lines.append("")
        for r in rows:
            ca, cb = concepts.get(r["cluster_a"]), concepts.get(r["cluster_b"])
            name_a = ca["concept_name"] if ca else r["cluster_a"]
            name_b = cb["concept_name"] if cb else r["cluster_b"]
            lines.append(
                f"- **{r['cluster_a']}** ({name_a}) <-> **{r['cluster_b']}** ({name_b}) "
                f"[{r['domain_pair'][0]}<->{r['domain_pair'][1]}, confidence={r['confidence']}]  \n"
                f"  _{r['evidence']}_"
            )
        lines.append("")

    out_path = taxonomy_dir / "SEMANTIC_HIERARCHY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[render_hierarchy_markdown] wrote {out_path}")


if __name__ == "__main__":
    main()
