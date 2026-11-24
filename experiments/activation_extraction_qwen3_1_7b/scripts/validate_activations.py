#!/usr/bin/env python
"""
Milestone 4, step 4 -- independent post-hoc validation of the full
activation dataset. Does NOT re-run the model. Checks all 12 items the
task requires:

  1. every expected prompt accounted for
  2. every retained token has one activation vector
  3. no padding tokens included (cross-checked against independent re-tokenization)
  4. no prompt has duplicate activation rows
  5. activation dimensions consistent
  6. activation dtype matches config
  7. no NaN/Inf
  8. prompt_ids and token positions traceable
  9. shard manifests match actual files
  10. train/validation split integrity (no prompt split across both)
  11. all five domains represented
  12. raw/normalized corpus files unchanged (git diff)

Usage:
    python validate_activations.py
"""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_common as ec  # noqa: E402


def main():
    exp_dir = ec.EXP_DIR
    metadata_dir = exp_dir / "metadata"
    errors = []
    warnings = []

    records = ec.load_normalized_records()
    manifest = json.loads((metadata_dir / "shard_manifest.json").read_text(encoding="utf-8"))
    meta_rows = [json.loads(l) for l in (metadata_dir / "prompt_token_metadata.jsonl").open(encoding="utf-8")]
    split_rows = [json.loads(l) for l in (metadata_dir / "prompt_split.jsonl").open(encoding="utf-8")]
    model_manifest = json.loads((metadata_dir / "model_manifest.json").read_text(encoding="utf-8"))
    hidden_size = model_manifest["architecture"]["hidden_size"]

    # 1. every expected prompt accounted for
    meta_pids = set(m["prompt_id"] for m in meta_rows)
    normalized_pids = set(records.keys())
    missing = normalized_pids - meta_pids
    extra = meta_pids - normalized_pids
    if missing:
        errors.append(f"{len(missing)} normalized prompts missing from activation metadata (e.g. {list(missing)[:5]})")
    if extra:
        errors.append(f"{len(extra)} activation prompt_ids not in normalized corpus (e.g. {list(extra)[:5]})")

    # 2 & 3. every retained token has exactly one row, no padding included (re-tokenize independently)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model_utils as mu
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mu.MODEL_NAME, cache_dir=str(mu.WEIGHTS_CACHE_DIR))

    rows_per_prompt = defaultdict(list)
    for m in meta_rows:
        rows_per_prompt[m["prompt_id"]].append(m)

    token_count_mismatches = []
    position_gap_or_dup = []
    for pid, rows in rows_per_prompt.items():
        true_len = len(tok(records[pid]["prompt"])["input_ids"])
        if len(rows) != true_len:
            token_count_mismatches.append({"prompt_id": pid, "stored": len(rows), "true": true_len})
        positions = sorted(r["token_position"] for r in rows)
        if positions != list(range(len(rows))):
            position_gap_or_dup.append(pid)
    if token_count_mismatches:
        errors.append(f"{len(token_count_mismatches)} prompts have a token count mismatch vs. independent "
                      f"re-tokenization (e.g. {token_count_mismatches[:3]}) -- may indicate padding leakage "
                      f"or dropped tokens")
    if position_gap_or_dup:
        errors.append(f"{len(position_gap_or_dup)} prompts have non-contiguous/duplicate token_position "
                      f"values (e.g. {position_gap_or_dup[:5]})")

    # 4. no duplicate activation rows (same prompt_id + token_position + shard/row twice)
    seen = set()
    duplicates = []
    for m in meta_rows:
        key = (m["prompt_id"], m["token_position"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate (prompt_id, token_position) metadata rows found "
                      f"(e.g. {duplicates[:5]})")

    # 5, 6, 7, 9: shard-level checks
    from safetensors.torch import load_file
    dim_mismatches, dtype_mismatches, nonfinite_shards, sha_mismatches = [], [], [], []
    total_rows_on_disk = 0
    for split in ("train", "validation"):
        for s in manifest.get(split, {}).get("shards", []):
            if not ec.validate_shard_on_disk(exp_dir, split, s):
                sha_mismatches.append(s["file"])
                continue
            path = exp_dir / "activations" / split / Path(s["file"]).name
            t = load_file(str(path))["activations"]
            if t.shape[1] != hidden_size:
                dim_mismatches.append(s["file"])
            if t.dtype != torch.bfloat16:
                dtype_mismatches.append(s["file"])
            if not torch.isfinite(t.float()).all():
                nonfinite_shards.append(s["file"])
            if t.shape[0] != s["n_rows"]:
                errors.append(f"shard {s['file']}: on-disk rows ({t.shape[0]}) != manifest n_rows ({s['n_rows']})")
            total_rows_on_disk += t.shape[0]

    if sha_mismatches:
        errors.append(f"{len(sha_mismatches)} shards failed sha256 verification: {sha_mismatches}")
    if dim_mismatches:
        errors.append(f"{len(dim_mismatches)} shards have wrong activation dimension: {dim_mismatches}")
    if dtype_mismatches:
        errors.append(f"{len(dtype_mismatches)} shards are not bfloat16: {dtype_mismatches}")
    if nonfinite_shards:
        errors.append(f"{len(nonfinite_shards)} shards contain non-finite values: {nonfinite_shards}")
    if total_rows_on_disk != len(meta_rows):
        errors.append(f"total on-disk rows ({total_rows_on_disk}) != total metadata rows ({len(meta_rows)})")

    # 8. prompt_ids and token positions traceable: every meta row has a valid shard_id/row_in_shard
    shard_keys = set()
    for split in ("train", "validation"):
        for s in manifest.get(split, {}).get("shards", []):
            shard_keys.add((split, s["shard_id"]))
    untraceable = [m for m in meta_rows if (m["split"], m["shard_id"]) not in shard_keys]
    if untraceable:
        errors.append(f"{len(untraceable)} metadata rows reference a shard not in the manifest")

    # 10. train/validation split integrity: a prompt's rows must all have the same split,
    # and that split must match prompt_split.jsonl
    split_of_prompt = {r["prompt_id"]: r["split"] for r in split_rows}
    split_violations = []
    for pid, rows in rows_per_prompt.items():
        splits_seen = set(r["split"] for r in rows)
        if len(splits_seen) > 1:
            split_violations.append({"prompt_id": pid, "splits": list(splits_seen)})
        elif pid in split_of_prompt and splits_seen != {split_of_prompt[pid]}:
            split_violations.append({"prompt_id": pid, "meta_split": list(splits_seen),
                                       "assigned_split": split_of_prompt[pid]})
    if split_violations:
        errors.append(f"{len(split_violations)} prompts have split inconsistencies (e.g. {split_violations[:5]})")

    # 11. all five domains represented
    domains_present = set(records[pid]["domain"] for pid in meta_pids if pid in records)
    missing_domains = set(ec.DOMAINS) - domains_present
    if missing_domains:
        errors.append(f"domains missing from activation data: {missing_domains}")

    # 12. source corpus unchanged
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", "normalized", "raw", "processed", "MI_Dataset_validated"],
            cwd=str(ec.ROOT), capture_output=True, text=True, timeout=30,
        )
        source_unchanged = diff.stdout.strip() == "" and diff.returncode == 0
        if not source_unchanged:
            errors.append(f"source corpus appears modified: git diff --stat output: {diff.stdout[:500]}")
    except Exception as e:
        warnings.append(f"could not run git diff check: {e}")
        source_unchanged = None

    result = {
        "n_normalized_prompts": len(normalized_pids),
        "n_prompts_in_activations": len(meta_pids),
        "n_metadata_rows": len(meta_rows),
        "n_rows_on_disk": total_rows_on_disk,
        "n_shards": sum(len(manifest.get(s, {}).get("shards", [])) for s in ("train", "validation")),
        "domains_present": sorted(domains_present),
        "source_corpus_unchanged": source_unchanged,
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }
    print(json.dumps(result, indent=2))

    out_path = exp_dir / "reports" / "activation_validation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[validate] wrote {out_path}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
