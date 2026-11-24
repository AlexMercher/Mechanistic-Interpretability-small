#!/usr/bin/env python
"""
Exact-duplicate multiplicity audit over the normalized 25K corpus.

Reads normalized/*.jsonl READ-ONLY. For every record, recomputes
sha256(prompt) independently of the stored `prompt_sha256` field and uses
the recomputed hash (not the stored one) as the grouping key -- this is
the "independently verify the hash against the actual prompt text" step.
Any record where stored != recomputed is reported as a conflict, not
silently trusted.

Does not delete, merge, or rewrite anything. Safe to re-run: it only
(re)writes files under its own --out-dir.

Usage:
    python scripts/audit_exact_duplicates.py [--normalized-dir normalized] [--out-dir experiments/duplicate_language_embedding/reports] [--overwrite]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NORMALIZED_DIR = ROOT / "normalized"
DEFAULT_OUT_DIR = ROOT / "experiments" / "duplicate_language_embedding" / "reports"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_all_records(normalized_dir: Path):
    records = []
    for jsonl_path in sorted(normalized_dir.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                rec = json.loads(line)
                rec["_source_file"] = jsonl_path.name
                rec["_source_line"] = lineno
                records.append(rec)
    return records


def check_overwrite(out_dir: Path, overwrite: bool, filenames):
    existing = [name for name in filenames if (out_dir / name).exists()]
    if existing and not overwrite:
        print(f"ERROR: outputs already exist in {out_dir}: {existing}", file=sys.stderr)
        print("Pass --overwrite to regenerate them.", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing report files")
    args = ap.parse_args()

    out_names = [
        "duplicate_metrics.json", "exact_duplicate_groups.csv",
        "duplicate_multiplicity_distribution.csv", "duplicate_domain_breakdown.csv",
        "duplicate_metadata_conflicts.csv",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    check_overwrite(args.out_dir, args.overwrite, out_names)

    print(f"[audit_exact_duplicates] loading records from {args.normalized_dir} ...")
    records = load_all_records(args.normalized_dir)
    total_records = len(records)
    print(f"[audit_exact_duplicates] loaded {total_records} records")

    # --- independent hash verification ---
    hash_mismatches = []
    verified_hash_by_id = {}
    for rec in records:
        recomputed = sha256_text(rec["prompt"])
        verified_hash_by_id[rec["prompt_id"]] = recomputed
        stored = rec.get("prompt_sha256")
        if stored != recomputed:
            hash_mismatches.append({
                "prompt_id": rec["prompt_id"], "domain": rec["domain"],
                "stored_prompt_sha256": stored, "recomputed_prompt_sha256": recomputed,
            })
    print(f"[audit_exact_duplicates] stored-vs-recomputed hash mismatches: {len(hash_mismatches)}")

    # --- group by the INDEPENDENTLY RECOMPUTED hash (ground truth), not the stored field ---
    groups = defaultdict(list)
    for rec in records:
        groups[verified_hash_by_id[rec["prompt_id"]]].append(rec)

    unique_prompt_texts = len(groups)
    duplicate_groups = {h: g for h, g in groups.items() if len(g) > 1}
    duplicate_group_count = len(duplicate_groups)
    duplicate_record_count = sum(len(g) for g in duplicate_groups.values())
    duplicate_excess_count = duplicate_record_count - duplicate_group_count
    max_multiplicity = max((len(g) for g in groups.values()), default=0)

    multiplicity_distribution = defaultdict(int)  # multiplicity -> group_count
    for g in groups.values():
        multiplicity_distribution[len(g)] += 1

    # --- within-text-group metadata conflicts & cross-domain groups ---
    conflicts = list({
        "conflict_type": "stored_hash_mismatch",
        "prompt_sha256": m["recomputed_prompt_sha256"],
        "prompt_id": m["prompt_id"], "domain": m["domain"], "subdomain": "", "category": "",
        "detail": f"stored={m['stored_prompt_sha256']} recomputed={m['recomputed_prompt_sha256']}",
    } for m in hash_mismatches)

    cross_domain_group_hashes = []
    for h, g in duplicate_groups.items():
        domains = {r["domain"] for r in g}
        subdomains = {r["subdomain"] for r in g}
        categories = {r["category"] for r in g}
        if len(domains) > 1:
            cross_domain_group_hashes.append(h)
            for r in g:
                conflicts.append({
                    "conflict_type": "cross_domain_duplicate",
                    "prompt_sha256": h, "prompt_id": r["prompt_id"], "domain": r["domain"],
                    "subdomain": r["subdomain"], "category": r["category"],
                    "detail": f"group spans domains: {sorted(domains)}",
                })
        if len(subdomains) > 1 or len(categories) > 1:
            for r in g:
                conflicts.append({
                    "conflict_type": "metadata_divergence",
                    "prompt_sha256": h, "prompt_id": r["prompt_id"], "domain": r["domain"],
                    "subdomain": r["subdomain"], "category": r["category"],
                    "detail": f"group subdomains={sorted(subdomains)} categories={sorted(categories)}",
                })

    # --- breakdown by domain / subdomain / category (duplicate groups only) ---
    # Rows at three granularities in one file: domain-only ("ALL" subdomain/category),
    # domain+subdomain ("ALL" category), and domain+subdomain+category (finest).
    # A group whose members span multiple domains/subdomains/categories is counted once
    # per distinct value it touches at that granularity (documented in FINAL_REPORT.md).
    breakdown_counts = defaultdict(lambda: {"duplicate_group_count": 0, "duplicate_record_count": 0})

    def add_breakdown(key, group_contrib, record_contrib):
        breakdown_counts[key]["duplicate_group_count"] += group_contrib
        breakdown_counts[key]["duplicate_record_count"] += record_contrib

    for h, g in duplicate_groups.items():
        n = len(g)
        domains = sorted({r["domain"] for r in g})
        for d in domains:
            add_breakdown((d, "ALL", "ALL"), 1, sum(1 for r in g if r["domain"] == d))
        dom_subdom = sorted({(r["domain"], r["subdomain"]) for r in g})
        for d, s in dom_subdom:
            add_breakdown((d, s, "ALL"), 1, sum(1 for r in g if r["domain"] == d and r["subdomain"] == s))
        dom_sub_cat = sorted({(r["domain"], r["subdomain"], r["category"]) for r in g})
        for d, s, c in dom_sub_cat:
            add_breakdown((d, s, c), 1, sum(
                1 for r in g if r["domain"] == d and r["subdomain"] == s and r["category"] == c
            ))

    # --- write outputs ---
    metrics = {
        "total_records": total_records,
        "unique_prompt_texts": unique_prompt_texts,
        "duplicate_group_count": duplicate_group_count,
        "duplicate_record_count": duplicate_record_count,
        "duplicate_excess_count": duplicate_excess_count,
        "max_multiplicity": max_multiplicity,
        "stored_hash_mismatch_count": len(hash_mismatches),
        "cross_domain_duplicate_group_count": len(cross_domain_group_hashes),
        "metadata_divergence_conflict_count": sum(
            1 for c in conflicts if c["conflict_type"] == "metadata_divergence"
        ),
        "reconciliation": {
            "formula": "total_records == unique_prompt_texts + duplicate_excess_count",
            "total_records": total_records,
            "unique_prompt_texts_plus_excess": unique_prompt_texts + duplicate_excess_count,
            "reconciles": total_records == unique_prompt_texts + duplicate_excess_count,
        },
        "hash_verification_method": "sha256(prompt_text_utf8), recomputed independently and used "
                                     "as the grouping key; stored prompt_sha256 field is only "
                                     "cross-checked against it, never trusted blindly",
    }
    (args.out_dir / "duplicate_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (args.out_dir / "exact_duplicate_groups.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_sha256", "multiplicity", "prompt_ids", "domains", "subdomains",
                          "categories", "prompt_text"])
        for h, g in sorted(duplicate_groups.items(), key=lambda kv: len(kv[1]), reverse=True):
            writer.writerow([
                h, len(g),
                ";".join(r["prompt_id"] for r in g),
                ";".join(sorted({r["domain"] for r in g})),
                ";".join(sorted({r["subdomain"] for r in g})),
                ";".join(sorted({r["category"] for r in g})),
                g[0]["prompt"],
            ])

    with (args.out_dir / "duplicate_multiplicity_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["multiplicity", "group_count", "total_records_at_this_multiplicity"])
        for mult in sorted(multiplicity_distribution):
            gc = multiplicity_distribution[mult]
            writer.writerow([mult, gc, mult * gc])

    with (args.out_dir / "duplicate_domain_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "subdomain", "category", "duplicate_group_count",
                          "duplicate_record_count", "duplicate_excess_count"])
        for (d, s, c), counts in sorted(breakdown_counts.items()):
            writer.writerow([d, s, c, counts["duplicate_group_count"], counts["duplicate_record_count"],
                              counts["duplicate_record_count"] - counts["duplicate_group_count"]])

    with (args.out_dir / "duplicate_metadata_conflicts.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["conflict_type", "prompt_sha256", "prompt_id", "domain", "subdomain", "category", "detail"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(conflicts)

    print(f"[audit_exact_duplicates] total_records={total_records} unique_prompt_texts={unique_prompt_texts} "
          f"duplicate_group_count={duplicate_group_count} duplicate_record_count={duplicate_record_count} "
          f"duplicate_excess_count={duplicate_excess_count} max_multiplicity={max_multiplicity}")
    print(f"[audit_exact_duplicates] reconciles={metrics['reconciliation']['reconciles']}")
    print(f"[audit_exact_duplicates] stored_hash_mismatch_count={len(hash_mismatches)} "
          f"cross_domain_duplicate_group_count={len(cross_domain_group_hashes)}")
    print(f"[audit_exact_duplicates] wrote reports to {args.out_dir}")

    if not metrics["reconciliation"]["reconciles"]:
        print("ERROR: duplicate counts do not reconcile mathematically!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
