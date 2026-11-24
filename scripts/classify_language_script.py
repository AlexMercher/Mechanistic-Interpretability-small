#!/usr/bin/env python
"""
Language / script / language-form classification for every record in the
normalized 25K corpus. Read-only against normalized/*.jsonl -- writes a
separate derived annotation dataset, never touches the source prompt text.

See scripts/lang_script_utils.py for the deterministic classification
method and its documented limitations.

Usage:
    python scripts/classify_language_script.py [--normalized-dir normalized] [--out-dir experiments/duplicate_language_embedding] [--overwrite]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_script_utils import classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NORMALIZED_DIR = ROOT / "normalized"
DEFAULT_OUT_DIR = ROOT / "experiments" / "duplicate_language_embedding"

VALID_LANGUAGES = {"Hindi", "English", "mixed", "other", "uncertain"}
VALID_SCRIPTS = {"Devanagari", "Latin", "mixed", "other", "none", "uncertain"}
VALID_LANGUAGE_FORMS = {
    "native_hindi", "romanized_hindi", "hinglish", "english",
    "mixed_multilingual", "other", "uncertain",
}


def load_all_records(normalized_dir: Path):
    records = []
    for jsonl_path in sorted(normalized_dir.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    annotations_dir = args.out_dir / "annotations"
    reports_dir = args.out_dir / "reports"
    annotations_path = annotations_dir / "language_script_annotations.jsonl"

    if annotations_path.exists() and not args.overwrite:
        print(f"ERROR: {annotations_path} already exists. Pass --overwrite to regenerate.", file=sys.stderr)
        sys.exit(1)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[classify_language_script] loading records from {args.normalized_dir} ...")
    records = load_all_records(args.normalized_dir)
    print(f"[classify_language_script] loaded {len(records)} records")

    domain_breakdown = Counter()
    review_rows = []
    invalid_rows = []
    n_review = 0

    with annotations_path.open("w", encoding="utf-8") as f:
        for rec in records:
            result = classify(rec["prompt"])

            if result["language"] not in VALID_LANGUAGES:
                invalid_rows.append((rec["prompt_id"], "language", result["language"]))
            if result["script"] not in VALID_SCRIPTS:
                invalid_rows.append((rec["prompt_id"], "script", result["script"]))
            if result["language_form"] not in VALID_LANGUAGE_FORMS:
                invalid_rows.append((rec["prompt_id"], "language_form", result["language_form"]))

            annotation = {
                "prompt_id": rec["prompt_id"],
                "domain": rec["domain"],
                "subdomain": rec["subdomain"],
                "category": rec["category"],
                "language": result["language"],
                "script": result["script"],
                "language_form": result["language_form"],
                "classification_method": result["classification_method"],
                "classification_confidence": result["classification_confidence"],
                "requires_manual_review": result["requires_manual_review"],
                "debug": result["debug"],
            }
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

            domain_breakdown[(rec["domain"], result["language"], result["script"], result["language_form"])] += 1

            if result["requires_manual_review"]:
                n_review += 1
                review_rows.append({
                    "prompt_id": rec["prompt_id"], "domain": rec["domain"],
                    "language": result["language"], "script": result["script"],
                    "language_form": result["language_form"],
                    "classification_confidence": result["classification_confidence"],
                    "prompt_snippet": rec["prompt"][:120],
                })

    if invalid_rows:
        print(f"ERROR: {len(invalid_rows)} records produced out-of-schema controlled values: "
              f"{invalid_rows[:10]}", file=sys.stderr)
        sys.exit(1)

    with (reports_dir / "language_script_domain_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "language", "script", "language_form", "count"])
        for (domain, lang, script, form), count in sorted(domain_breakdown.items()):
            writer.writerow([domain, lang, script, form, count])

    with (reports_dir / "language_script_review.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["prompt_id", "domain", "language", "script", "language_form",
                      "classification_confidence", "prompt_snippet"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    lang_totals = Counter()
    script_totals = Counter()
    form_totals = Counter()
    per_domain_totals = defaultdict(Counter)
    for (domain, lang, script, form), count in domain_breakdown.items():
        lang_totals[lang] += count
        script_totals[script] += count
        form_totals[form] += count
        per_domain_totals[domain][lang] += count

    metrics = {
        "total_records": len(records),
        "language_totals": dict(lang_totals),
        "script_totals": dict(script_totals),
        "language_form_totals": dict(form_totals),
        "requires_manual_review_count": n_review,
        "requires_manual_review_fraction": round(n_review / len(records), 4) if records else 0.0,
        "language_by_domain": {d: dict(c) for d, c in per_domain_totals.items()},
        "classification_method": "deterministic_unicode_script_plus_lexical_markers_v1",
        "ml_language_id_model_used": False,
        "ml_language_id_model_rationale": "deterministic Unicode-script + closed-class lexical-marker "
            "rules were judged sufficient for this templated, limited-vocabulary corpus; see "
            "scripts/lang_script_utils.py module docstring and FINAL_REPORT.md for full rationale "
            "and documented limitations (romanized_hindi vs hinglish boundary is approximate).",
    }
    (reports_dir / "language_script_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[classify_language_script] wrote {len(records)} annotations -> {annotations_path}")
    print(f"[classify_language_script] requires_manual_review: {n_review} "
          f"({metrics['requires_manual_review_fraction']:.2%})")
    print(json.dumps({"language_totals": dict(lang_totals), "script_totals": dict(script_totals)}, indent=2))


if __name__ == "__main__":
    main()
