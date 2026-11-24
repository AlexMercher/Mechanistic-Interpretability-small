"""
Milestone 6, Phase 3 pre-step (plan B.5): deterministic, stratified sample of EXACTLY
40 candidates for the blind second-reviewer pass, proportional to each candidate's
primary_stratum via the largest-remainder method.

Written BEFORE any review happens, per the plan -- the sample composition cannot be
influenced by review outcomes.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

SFD_DIR = Path(__file__).resolve().parent.parent
TARGET_TOTAL = 40


def main():
    candidates = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8")]
    by_stratum = defaultdict(list)
    for c in candidates:
        by_stratum[c["primary_stratum"]].append(c["feature_idx"])
    for feats in by_stratum.values():
        feats.sort()

    total = len(candidates)
    exact_shares = {s: len(feats) * TARGET_TOTAL / total for s, feats in by_stratum.items()}
    floors = {s: math.floor(v) for s, v in exact_shares.items()}
    remainders = {s: exact_shares[s] - floors[s] for s in exact_shares}

    n_allocated = sum(floors.values())
    n_remaining = TARGET_TOTAL - n_allocated
    # Largest-remainder method; ties broken deterministically by stratum name (ascending).
    order = sorted(remainders.keys(), key=lambda s: (-remainders[s], s))
    counts = dict(floors)
    for s in order[:n_remaining]:
        counts[s] += 1

    assert sum(counts.values()) == TARGET_TOTAL, f"allocation sums to {sum(counts.values())}, expected {TARGET_TOTAL}"

    sample = []
    for stratum, n in counts.items():
        chosen = by_stratum[stratum][:n]  # deterministic: ascending feature_idx within stratum
        sample.extend(chosen)

    sample = sorted(sample)
    config = {
        "method": "largest_remainder_proportional_to_primary_stratum",
        "target_total": TARGET_TOTAL,
        "tie_break_rule_for_remainders": "ascending stratum name",
        "tie_break_rule_for_within_stratum_selection": "ascending feature_idx",
        "per_stratum_pool_size": {s: len(f) for s, f in by_stratum.items()},
        "per_stratum_exact_share": exact_shares,
        "per_stratum_allocated_count": counts,
        "total_allocated": sum(counts.values()),
        "sampled_feature_indices": sample,
    }
    out_path = SFD_DIR / "configs" / "second_reviewer_sample.json"
    out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "sampled_feature_indices"}, indent=2))
    print(f"[sample] {len(sample)} candidates: {sample}")
    print(f"[sample] wrote {out_path}")


if __name__ == "__main__":
    main()
