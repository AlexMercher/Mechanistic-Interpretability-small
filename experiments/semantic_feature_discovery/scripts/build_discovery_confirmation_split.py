"""
Milestone 6, Phase 1 step 1 (plan B.3): exact, rank-based, per-domain deterministic
partition of the existing frozen Milestone-4 validation split (2,533 prompts) into
Discovery and Confirmation subsets.

This is a NEW partition owned entirely by Milestone 6 -- it does not alter
`prompt_split.jsonl`, the activation shards, or any other M4/M5 artifact. It exists
to prevent the same validation data from being used to both select candidate
features and confirm their interpretations (a selection-bias risk raised in review).

Method (frozen, not adjustable after seeing results):
  1. Group the 2,533 validation prompt_ids by domain (5 groups).
  2. Within each domain, h(prompt_id) = sha256(f"m6_discovery_confirmation:{prompt_id}")
     (hex digest).
  3. Sort the domain's prompt_ids by h(prompt_id) ascending (ties broken by prompt_id
     string -- sha256 collisions are not expected to occur in practice).
  4. First floor(n/2) (sorted order) -> Discovery; remaining ceil(n/2) -> Confirmation.
     The extra prompt in an odd-sized domain group goes to Confirmation.

Verification (part of this script's own output, not a separate manual step):
  - every validation prompt_id appears in exactly one subset
  - per-domain counts are exactly floor(n/2)/ceil(n/2)
  - every token in prompt_token_metadata.jsonl inherits its prompt's subset,
    checked exhaustively (every row, not sampled)
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
HASH_NAMESPACE = "m6_discovery_confirmation"


def main():
    records = sf.load_normalized_records()
    validation_ids = sf.load_validation_prompt_ids()

    by_domain = defaultdict(list)
    for pid in validation_ids:
        domain = records[pid]["domain"]
        by_domain[domain].append(pid)

    assignment = {}
    per_domain_counts = {}
    for domain, pids in sorted(by_domain.items()):
        hashed = sorted(pids, key=lambda p: (hashlib.sha256(f"{HASH_NAMESPACE}:{p}".encode()).hexdigest(), p))
        n = len(hashed)
        n_discovery = n // 2
        discovery_ids = hashed[:n_discovery]
        confirmation_ids = hashed[n_discovery:]
        for pid in discovery_ids:
            assignment[pid] = "discovery"
        for pid in confirmation_ids:
            assignment[pid] = "confirmation"
        per_domain_counts[domain] = {
            "total": n, "discovery": len(discovery_ids), "confirmation": len(confirmation_ids),
        }
        assert len(discovery_ids) == n // 2
        assert len(confirmation_ids) == n - n // 2

    # Verification 1: every validation prompt_id assigned exactly once
    assert set(assignment.keys()) == set(validation_ids), "prompt coverage mismatch"
    assert len(assignment) == len(validation_ids), "duplicate assignment detected"

    # Verification 2: exhaustive token-inheritance check against prompt_token_metadata.jsonl
    token_mismatches = []
    n_validation_tokens_checked = 0
    with sf.PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != "validation":
                continue
            n_validation_tokens_checked += 1
            pid = row["prompt_id"]
            if pid not in assignment:
                token_mismatches.append(pid)

    if token_mismatches:
        raise RuntimeError(f"{len(set(token_mismatches))} validation prompt_ids in "
                            f"prompt_token_metadata.jsonl have no Discovery/Confirmation "
                            f"assignment -- refusing to proceed")

    out_dir = SFD_DIR / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "exact_rank_based_per_domain_sha256_split",
        "hash_namespace": HASH_NAMESPACE,
        "rounding_rule": "floor(n/2) -> discovery, ceil(n/2) -> confirmation (extra prompt in odd-sized domain groups goes to confirmation)",
        "per_domain_counts": per_domain_counts,
        "total_validation_prompts": len(validation_ids),
        "total_discovery_prompts": sum(v["discovery"] for v in per_domain_counts.values()),
        "total_confirmation_prompts": sum(v["confirmation"] for v in per_domain_counts.values()),
        "n_validation_tokens_checked_for_inheritance": n_validation_tokens_checked,
        "token_inheritance_check_passed": True,
        "verified_every_prompt_assigned_exactly_once": True,
    }
    (out_dir / "discovery_confirmation_split_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    mapping_path = out_dir / "discovery_confirmation_assignment.jsonl"
    with mapping_path.open("w", encoding="utf-8") as f:
        for pid in sorted(assignment.keys()):
            f.write(json.dumps({"prompt_id": pid, "subset": assignment[pid],
                                 "domain": records[pid]["domain"]}) + "\n")

    print(json.dumps(config, indent=2))
    print(f"[split] wrote {out_dir / 'discovery_confirmation_split_config.json'}")
    print(f"[split] wrote {mapping_path}")


if __name__ == "__main__":
    main()
