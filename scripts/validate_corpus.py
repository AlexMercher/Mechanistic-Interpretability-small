"""
Step 2 validation of the 25K normalized corpus, run before any semantic
processing. Pure stdlib (json, csv, re, difflib, hashlib) -- no embeddings,
no deletion. Near-duplicates are *flagged for review*, never auto-removed:
two prompts can share surface form but probe different reasoning patterns.

Writes all artifacts under experiments/corpus_validation_25k/.
Run from the project root, after the build_* scripts: python scripts/validate_corpus.py
"""
import csv
import json
import re
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"
PROCESSED_DIR = ROOT / "processed"
EXPERIMENT_DIR = ROOT / "experiments" / "corpus_validation_25k"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

REQUIRED_FIELDS = ["domain", "subdomain", "prompt", "source_model", "source_file"]
DEVANAGARI_RANGE = (0x0900, 0x097F)
NEAR_DUP_THRESHOLD = 0.72
LENGTH_OUTLIER_WORDS = 120  # absolute word-count ceiling flagged for review


def load_domain(jsonl_path: Path):
    with jsonl_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def has_devanagari(text: str) -> bool:
    return any(DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1] for ch in text)


def check_record(domain: str, rec: dict):
    """Returns (hard_issues, soft_flags) for one record.

    hard_issues: structural defects -- missing/blank required data, actual mojibake
    (U+FFFD). These make a record invalid.

    soft_flags: content characteristics worth a human look but not a defect -- e.g.
    Romanized/Hinglish HINDI prompts with no Devanagari codepoints. Confirmed by
    manual inspection (see EXPERIMENT.md) to be legitimate transliterated prompts
    from all three source generators, not corruption, so they are NOT counted as
    invalid.
    """
    hard_issues = []
    soft_flags = []
    for field in REQUIRED_FIELDS:
        if not rec.get(field):
            hard_issues.append(f"missing_or_empty:{field}")
    prompt = rec.get("prompt", "")
    if not prompt.strip():
        hard_issues.append("blank_prompt")
    if "�" in prompt:
        hard_issues.append("encoding_replacement_char")
    if domain == "HINDI" and prompt.strip() and not has_devanagari(prompt):
        soft_flags.append("hindi_romanized_no_devanagari")
    return hard_issues, soft_flags


def check_id_sequence(domain: str, records: list):
    """Expects prompt_id = f'{domain}_{seq:06d}' covering 1..len(records) with no gaps/dupes."""
    seqs = []
    for rec in records:
        m = re.match(rf"^{domain}_(\d{{6}})$", rec["prompt_id"])
        if not m:
            continue
        seqs.append(int(m.group(1)))
    expected = set(range(1, len(records) + 1))
    actual = set(seqs)
    missing = sorted(expected - actual)
    duplicates = len(seqs) - len(actual)
    return {"missing_ids": missing, "duplicate_count": duplicates}


def near_duplicate_pairs(records: list):
    """Blocks by (category, subdomain) -- template reuse is localized there (confirmed
    by exact-duplicate inspection), keeping pairwise comparison tractable. Uses difflib's
    cheap quick_ratio/real_quick_ratio upper bounds to skip the expensive ratio() call."""
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["category"], rec["subdomain"])].append(rec)

    candidates = []
    for (category, subdomain), group in groups.items():
        n = len(group)
        for i in range(n):
            a = group[i]
            sm = SequenceMatcher(None, a["prompt"])
            for j in range(i + 1, n):
                b = group[j]
                if a["prompt_sha256"] == b["prompt_sha256"]:
                    continue  # exact duplicates are reported separately
                sm.set_seq2(b["prompt"])
                if sm.real_quick_ratio() < NEAR_DUP_THRESHOLD:
                    continue
                if sm.quick_ratio() < NEAR_DUP_THRESHOLD:
                    continue
                ratio = sm.ratio()
                if ratio >= NEAR_DUP_THRESHOLD:
                    candidates.append({
                        "prompt_id_a": a["prompt_id"],
                        "prompt_id_b": b["prompt_id"],
                        "category": category,
                        "subdomain": subdomain,
                        "similarity_ratio": round(ratio, 4),
                    })
    return candidates


def length_stats_and_outliers(domain: str, records: list):
    lengths = [len(rec["prompt"].split()) for rec in records]
    outliers = [
        {"prompt_id": rec["prompt_id"], "domain": domain, "word_count": len(rec["prompt"].split())}
        for rec in records
        if len(rec["prompt"].split()) > LENGTH_OUTLIER_WORDS
    ]
    stats = {
        "domain": domain,
        "n": len(lengths),
        "mean_words": round(statistics.mean(lengths), 2),
        "median_words": statistics.median(lengths),
        "p90_words": sorted(lengths)[int(0.9 * len(lengths)) - 1],
        "p99_words": sorted(lengths)[int(0.99 * len(lengths)) - 1],
        "max_words": max(lengths),
    }
    return stats, outliers


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    domains = {}
    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        domain = jsonl_path.stem.upper()
        domains[domain] = load_domain(jsonl_path)

    total_records = sum(len(r) for r in domains.values())

    invalid_rows = []
    flag_rows = []
    for domain, records in domains.items():
        for rec in records:
            hard_issues, soft_flags = check_record(domain, rec)
            if hard_issues:
                invalid_rows.append({
                    "prompt_id": rec["prompt_id"], "domain": domain, "issues": ";".join(hard_issues)
                })
            if soft_flags:
                flag_rows.append({
                    "prompt_id": rec["prompt_id"], "domain": domain, "flags": ";".join(soft_flags)
                })
    with (REPORTS_DIR / "invalid_records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "domain", "issues"])
        writer.writeheader()
        writer.writerows(invalid_rows)
    with (REPORTS_DIR / "review_flags.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "domain", "flags"])
        writer.writeheader()
        writer.writerows(flag_rows)

    id_sequence_report = {domain: check_id_sequence(domain, records) for domain, records in domains.items()}

    exact_dup_total = 0
    exact_dup_by_domain = {}
    for domain in domains:
        dup_csv = PROCESSED_DIR / "reports" / f"{domain.lower()}_exact_duplicates.csv"
        with dup_csv.open(encoding="utf-8") as f:
            n = sum(1 for _ in csv.DictReader(f))
        exact_dup_by_domain[domain] = n
        exact_dup_total += n

    all_near_dups = []
    near_dup_by_domain = {}
    for domain, records in domains.items():
        pairs = near_duplicate_pairs(records)
        near_dup_by_domain[domain] = len(pairs)
        for p in pairs:
            p["domain"] = domain
        all_near_dups.extend(pairs)
        print(f"{domain}: {len(pairs)} near-duplicate candidates (threshold={NEAR_DUP_THRESHOLD})")

    with (REPORTS_DIR / "near_duplicate_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["domain", "prompt_id_a", "prompt_id_b", "category", "subdomain", "similarity_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_near_dups)

    length_summaries = []
    all_outliers = []
    for domain, records in domains.items():
        stats, outliers = length_stats_and_outliers(domain, records)
        length_summaries.append(stats)
        all_outliers.extend(outliers)
    with (REPORTS_DIR / "length_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(length_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(length_summaries)
    with (REPORTS_DIR / "length_outliers.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "domain", "word_count"])
        writer.writeheader()
        writer.writerows(all_outliers)

    invalid_count = len(invalid_rows)
    flag_count = len(flag_rows)
    metrics = {
        "total_records": total_records,
        "valid_records": total_records - invalid_count,
        "invalid_records": invalid_count,
        "review_flags": flag_count,
        "review_flags_note": "content characteristics (e.g. Romanized Hindi with no Devanagari "
                              "codepoints) confirmed NOT to be corruption or defects -- not counted "
                              "against valid_records, listed for awareness only",
        "domain_counts": {d: len(r) for d, r in domains.items()},
        "id_sequence_check": id_sequence_report,
        "exact_duplicates": exact_dup_total,
        "exact_duplicates_by_domain": exact_dup_by_domain,
        "near_duplicate_candidates": len(all_near_dups),
        "near_duplicate_candidates_by_domain": near_dup_by_domain,
        "near_duplicate_similarity_threshold": NEAR_DUP_THRESHOLD,
        "length_outliers": len(all_outliers),
        "length_outlier_word_threshold": LENGTH_OUTLIER_WORDS,
        "length_stats_by_domain": length_summaries,
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({k: v for k, v in metrics.items() if k in (
        "total_records", "valid_records", "invalid_records", "review_flags",
        "exact_duplicates", "near_duplicate_candidates", "length_outliers"
    )}, indent=2))


if __name__ == "__main__":
    main()
