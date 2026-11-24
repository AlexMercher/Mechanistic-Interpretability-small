"""
Builds a tractable, representative sample of near-duplicate candidate pairs
(with full prompt text) for a teacher-model (Opus) adjudication pass.
Reviewing all ~36k candidate pairs individually isn't necessary -- this
samples the highest-similarity pairs per domain (where false "duplicates"
are most costly if ever acted on) plus a random cross-section, so the
adjudication is a calibration check on the heuristic, not an exhaustive audit.

Run from the project root: python scripts/sample_near_duplicates_for_review.py
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"
EXPERIMENT_DIR = ROOT / "experiments" / "corpus_validation_25k"
TOP_N_PER_DOMAIN = 15
RANDOM_N_PER_DOMAIN = 5
SEED = 0


def main():
    random.seed(SEED)
    prompts_by_id = {}
    for jsonl_path in NORMALIZED_DIR.glob("*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompts_by_id[rec["prompt_id"]] = rec["prompt"]

    rows_by_domain = {}
    with (EXPERIMENT_DIR / "reports" / "near_duplicate_candidates.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_domain.setdefault(row["domain"], []).append(row)

    sample = []
    for domain, rows in rows_by_domain.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["similarity_ratio"]), reverse=True)
        top = rows_sorted[:TOP_N_PER_DOMAIN]
        remaining = rows_sorted[TOP_N_PER_DOMAIN:]
        rand = random.sample(remaining, min(RANDOM_N_PER_DOMAIN, len(remaining)))
        for row in top + rand:
            sample.append({
                "domain": domain,
                "prompt_id_a": row["prompt_id_a"],
                "prompt_id_b": row["prompt_id_b"],
                "category": row["category"],
                "subdomain": row["subdomain"],
                "similarity_ratio": row["similarity_ratio"],
                "prompt_a": prompts_by_id[row["prompt_id_a"]],
                "prompt_b": prompts_by_id[row["prompt_id_b"]],
            })

    out_path = EXPERIMENT_DIR / "reports" / "near_duplicate_review_sample.json"
    out_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(sample)} pairs to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
