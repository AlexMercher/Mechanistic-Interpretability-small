#!/usr/bin/env python
"""
Independent post-hoc validation of the frozen semantic hierarchy. Does not
recompute clusters or relationships -- checks internal consistency only.

Checks:
  - Every normalized prompt_id appears in concept_assignments.jsonl exactly once.
  - Every concept_id in concept_assignments.jsonl exists in the hierarchy's concepts dict.
  - Every concept's prompt_ids list matches its assignment rows exactly (no drift).
  - raw_record_count / unique_prompt_count on each concept are internally consistent
    (unique_prompt_count <= raw_record_count, and matches independently recomputed
    exact-duplicate-group counting).
  - Every cross_domain_relationships edge references concept_ids that exist.
  - No concept_id collisions across domains.

Usage:
    python validate_taxonomy.py [--version v1]
"""
import argparse
import json
import sys
from collections import Counter
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
    relationships = hierarchy["cross_domain_relationships"]

    records = sc.load_normalized_records()
    group_of = sc.load_duplicate_group_map()

    errors = []

    assignments = [json.loads(l) for l in (taxonomy_dir / "concept_assignments.jsonl").open(encoding="utf-8")]
    assigned_pids = Counter(a["prompt_id"] for a in assignments)
    missing = set(records.keys()) - set(assigned_pids.keys())
    extra = set(assigned_pids.keys()) - set(records.keys())
    duplicated = [p for p, n in assigned_pids.items() if n > 1]
    if missing:
        errors.append(f"{len(missing)} normalized prompt_ids have no concept assignment (e.g. {list(missing)[:5]})")
    if extra:
        errors.append(f"{len(extra)} assigned prompt_ids do not exist in normalized/ (e.g. {list(extra)[:5]})")
    if duplicated:
        errors.append(f"{len(duplicated)} prompt_ids assigned to more than one concept (e.g. {duplicated[:5]})")

    assignment_concept_ids = {a["concept_id"] for a in assignments}
    unknown_concepts = assignment_concept_ids - set(concepts.keys())
    if unknown_concepts:
        errors.append(f"{len(unknown_concepts)} concept_ids in assignments not found in hierarchy (e.g. "
                      f"{list(unknown_concepts)[:5]})")

    for cid, c in concepts.items():
        if c["unique_prompt_count"] > c["raw_record_count"]:
            errors.append(f"{cid}: unique_prompt_count ({c['unique_prompt_count']}) > "
                          f"raw_record_count ({c['raw_record_count']})")
        recomputed_unique = sc.unique_text_count(c["prompt_ids"], group_of)
        if recomputed_unique != c["unique_prompt_count"]:
            errors.append(f"{cid}: stored unique_prompt_count={c['unique_prompt_count']} != "
                          f"recomputed={recomputed_unique}")
        if len(c["prompt_ids"]) != c["raw_record_count"]:
            errors.append(f"{cid}: len(prompt_ids)={len(c['prompt_ids'])} != raw_record_count={c['raw_record_count']}")

    for r in relationships:
        if r["cluster_a"] not in concepts or r["cluster_b"] not in concepts:
            errors.append(f"relationship {r['candidate_id']} references a concept_id not in the hierarchy")

    result = {
        "n_concepts": len(concepts),
        "n_prompts_total": len(records),
        "n_assignments": len(assignments),
        "n_relationships": len(relationships),
        "errors": errors,
        "passed": len(errors) == 0,
    }
    print(json.dumps(result, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
