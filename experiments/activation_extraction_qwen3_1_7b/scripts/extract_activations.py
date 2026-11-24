#!/usr/bin/env python
"""
Milestone 4, step 3 -- full 25,000-prompt residual-stream extraction.
Resumable: on restart, already-completed prompts (verified against
metadata/shard_manifest.json + on-disk shard sha256, not just existence)
are skipped automatically.

Usage:
    python extract_activations.py [--batch-size 8] [--rows-per-shard 100000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_common as ec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rows-per-shard", type=int, default=100000)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    exp_dir = ec.EXP_DIR
    metadata_dir = exp_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    records = ec.load_normalized_records()
    all_prompt_ids = sorted(records.keys())
    print(f"[extract] {len(all_prompt_ids)} total prompts in normalized corpus")

    split_assignments = ec.compute_prompt_split(all_prompt_ids)
    split_path = metadata_dir / "prompt_split.jsonl"
    if not split_path.exists():
        with split_path.open("w", encoding="utf-8") as f:
            for pid in all_prompt_ids:
                f.write(json.dumps({"prompt_id": pid, "split": split_assignments[pid],
                                     "domain": records[pid]["domain"]}, ensure_ascii=False) + "\n")
        print(f"[extract] wrote {split_path} (seed={ec.SPLIT_SEED}, val_fraction={ec.VAL_FRACTION})")
    else:
        # resumption: re-derive assignments from the already-written file so a restart
        # never silently uses a different split than the one already partially extracted
        existing = [json.loads(l) for l in split_path.open(encoding="utf-8")]
        split_assignments = {r["prompt_id"]: r["split"] for r in existing}
        print(f"[extract] reusing existing {split_path} ({len(existing)} prompts)")

    n_val = sum(1 for v in split_assignments.values() if v == "validation")
    print(f"[extract] split: {len(all_prompt_ids) - n_val} train, {n_val} validation")

    completed, manifest, _ = ec.load_existing_progress(metadata_dir)
    # load_existing_progress() only checked the manifest's own status field; independently
    # re-verify every shard's sha256 against the actual file on disk before trusting it,
    # in case a previous run was killed mid-write or a file was corrupted/moved.
    verified_completed = set(completed)
    for split in ("train", "validation"):
        for s in manifest.get(split, {}).get("shards", []):
            if not ec.validate_shard_on_disk(exp_dir, split, s):
                print(f"[extract] WARNING: shard {s['file']} failed sha256 verification -- "
                      f"its prompts will be treated as NOT completed and re-extracted.")
                for pid in s["prompt_ids"]:
                    verified_completed.discard(pid)

    remaining = [pid for pid in all_prompt_ids if pid not in verified_completed]
    print(f"[extract] {len(verified_completed)} prompts already completed (verified), "
          f"{len(remaining)} remaining")

    if not remaining:
        print("[extract] nothing to do -- all prompts already extracted and verified.")
        return

    t0 = time.time()
    stats = ec.run_extraction(
        remaining, records, split_assignments, exp_dir=exp_dir,
        batch_size=args.batch_size, rows_per_shard=args.rows_per_shard,
        dtype_name=args.dtype,
    )
    elapsed = time.time() - t0

    log_path = exp_dir / "logs" / f"extraction_run_{int(t0)}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({**stats, "batch_size": args.batch_size,
                                     "rows_per_shard": args.rows_per_shard,
                                     "dtype": args.dtype, "wall_clock_sec": elapsed},
                                    indent=2), encoding="utf-8")
    print(f"[extract] run complete in {elapsed:.1f}s. Log: {log_path}")
    print(json.dumps({k: v for k, v in stats.items() if k != "batch_times_sec"}, indent=2))

    if stats["failed_prompts"]:
        print(f"[extract] WARNING: {len(stats['failed_prompts'])} prompts failed -- "
              f"see log for details. Re-run this script to retry them.")


if __name__ == "__main__":
    main()
