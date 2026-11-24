"""
Shared helpers for the pilot and full-run activation extraction scripts.
Tokenizer behavior verified before writing this module: Qwen3's tokenizer
does NOT auto-add BOS/EOS tokens on plain `tokenizer(text)` calls (no
bos_token exists for this model at all -- see metadata/model_manifest.json
special_tokens.bos_token: null), so every id in a prompt's input_ids is a
real content token; only the padding added for batching needs to be
excluded via the attention mask.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_utils as mu  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent.parent
NORMALIZED_DIR = ROOT / "normalized"
EXP_DIR = ROOT / "experiments" / "activation_extraction_qwen3_1_7b"

DOMAINS = ["DNA", "EMOTIONS", "HINDI", "SCIENCE", "GANDHI"]

SPLIT_SEED = 1337
VAL_FRACTION = 0.1


def load_normalized_records():
    records = {}
    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["prompt_id"]] = rec
    return records


def compute_prompt_split(prompt_ids, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    """Deterministic prompt-level train/validation split. A prompt_id's split is
    a pure function of (prompt_id, seed, val_fraction) via a stable hash, so the
    split is identical across repeated runs and independent of iteration order."""
    assignments = {}
    for pid in prompt_ids:
        h = hashlib.sha256(f"{seed}:{pid}".encode("utf-8")).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        assignments[pid] = "validation" if frac < val_fraction else "train"
    return assignments


def build_batches(records, tokenizer, batch_size, max_length=None):
    """
    Length-aware batching: sorts prompt_ids by tokenized length so each batch
    pads only to the longest prompt WITHIN that batch, not the corpus-wide max.
    Returns a deterministic list of batches (list of prompt_id lists), ordering
    primarily by length then by prompt_id for full reproducibility.
    """
    lengths = {}
    for pid, rec in records.items():
        ids = tokenizer(rec["prompt"], add_special_tokens=True)["input_ids"]
        lengths[pid] = len(ids)

    ordered = sorted(records.keys(), key=lambda p: (lengths[p], p))
    batches = []
    for i in range(0, len(ordered), batch_size):
        batches.append(ordered[i:i + batch_size])
    return batches, lengths


class ShardWriter:
    """
    Writes one split's (train or validation) activation shards as
    safetensors files -- chosen over plain .npy because safetensors
    natively supports bfloat16 (numpy has no native bfloat16 dtype, so
    .npy would have required either a lossy fp16 downcast or a nonstandard
    uint16-bitcast workaround), supports fast memory-mapped/zero-copy reads
    via `safe_open` without loading the whole shard into RAM, and is the
    same format already used by this project's model weights (no new
    dependency). Each shard holds at most `rows_per_shard` token rows;
    a prompt's tokens are never split across two shards (a shard is
    flushed before accepting a prompt that would overflow it), which keeps
    resumption and validation simple (a shard is either fully written and
    valid, or does not exist yet).

    Shards are written atomically: to `<name>.tmp.safetensors` then
    os.rename()'d to the final name, so an interrupted write never leaves
    a corrupt file at the final path.
    """

    def __init__(self, split_dir: Path, split_name: str, rows_per_shard: int, existing_manifest: dict):
        import torch
        self.torch = torch
        self.split_dir = split_dir
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.rows_per_shard = rows_per_shard
        self.buffer_rows = []  # list of 1D tensors (one per token), moved to CPU
        self.buffer_meta = []  # list of dict (without activation_row_index/shard fields yet)
        existing_shards = existing_manifest.get(split_name, {}).get("shards", [])
        self.next_shard_id = max([s["shard_id"] for s in existing_shards], default=-1) + 1
        self.written_shards = []  # filled in during this run

    def add_prompt(self, token_rows, meta_rows):
        """token_rows: list[torch.Tensor] (1D, hidden_size, CPU, bfloat16). meta_rows: list[dict],
        same length, one per token, WITHOUT row_in_shard/shard_id/activation_row_index (filled on flush)."""
        assert len(token_rows) == len(meta_rows)
        if len(self.buffer_rows) + len(token_rows) > self.rows_per_shard and self.buffer_rows:
            self.flush()
        self.buffer_rows.extend(token_rows)
        self.buffer_meta.extend(meta_rows)
        if len(self.buffer_rows) >= self.rows_per_shard:
            self.flush()

    def flush(self):
        if not self.buffer_rows:
            return None
        from safetensors.torch import save_file
        shard_id = self.next_shard_id
        self.next_shard_id += 1
        stacked = self.torch.stack(self.buffer_rows, dim=0).contiguous()
        shard_name = f"shard_{shard_id:05d}.safetensors"
        final_path = self.split_dir / shard_name
        tmp_path = self.split_dir / f".tmp_{shard_name}"
        save_file({"activations": stacked}, str(tmp_path))
        tmp_path.replace(final_path)  # atomic rename on the same filesystem

        sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()
        for row_in_shard, meta in enumerate(self.buffer_meta):
            meta["shard_id"] = shard_id
            meta["row_in_shard"] = row_in_shard

        shard_record = {
            "shard_id": shard_id, "file": f"{self.split_name}/{shard_name}",
            "n_rows": stacked.shape[0], "sha256": sha256, "status": "complete",
            "prompt_ids": sorted({m["prompt_id"] for m in self.buffer_meta}),
        }
        self.written_shards.append((shard_record, list(self.buffer_meta)))
        self.buffer_rows = []
        self.buffer_meta = []
        return shard_record


class ResidualStreamHook:
    """Captures the output of a single decoder block (the residual stream
    AFTER that Transformer block) via a forward hook -- does NOT request
    output_hidden_states=True for the whole model, which would retain all
    28 layers' activations in memory simultaneously and waste scarce VRAM."""

    def __init__(self, model, layer_index):
        self.layer_index = layer_index
        self.captured = None
        target_module = model.model.layers[layer_index]
        self.handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # Qwen3 decoder layer forward returns a tuple; element 0 is the hidden state
        hidden = output[0] if isinstance(output, tuple) else output
        self.captured = hidden

    def remove(self):
        self.handle.remove()


def load_existing_progress(metadata_dir: Path):
    """Returns (completed_prompt_ids: set, shard_manifest: dict, existing_meta_rows: list)."""
    manifest_path = metadata_dir / "shard_manifest.json"
    meta_path = metadata_dir / "prompt_token_metadata.jsonl"
    manifest = {"train": {"shards": []}, "validation": {"shards": []}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    completed = set()
    existing_rows = []
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_rows.append(row)
    # a prompt is only "completed" if every token row it contributed belongs to a shard
    # marked complete in the manifest AND that shard file actually exists on disk and
    # matches its recorded sha256 -- verified by validate_shard() below, called by the caller.
    shard_status = {}
    for split in ("train", "validation"):
        for s in manifest.get(split, {}).get("shards", []):
            shard_status[(split, s["shard_id"])] = s

    prompt_to_shards = {}
    for row in existing_rows:
        key = (row["split"], row["shard_id"])
        prompt_to_shards.setdefault(row["prompt_id"], set()).add(key)

    for pid, keys in prompt_to_shards.items():
        if all(shard_status.get(k, {}).get("status") == "complete" for k in keys):
            completed.add(pid)

    return completed, manifest, existing_rows


def validate_shard_on_disk(exp_dir: Path, split: str, shard_record: dict) -> bool:
    path = exp_dir / "activations" / split / Path(shard_record["file"]).name
    if not path.exists():
        return False
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual_sha == shard_record["sha256"]


def extract_tokens_from_output(hidden_batch, attention_mask, input_ids, prompt_ids_in_batch,
                                records, annotations_lookup, split_assignments,
                                selected_layer, hook_location, dtype_name):
    """
    hidden_batch: (B, T, H) tensor on GPU. attention_mask/input_ids: (B, T) on GPU (or CPU).
    Returns dict[split] -> (token_rows: list[Tensor cpu], meta_rows: list[dict]) for this batch,
    excluding every padded position via attention_mask.
    """
    out = {"train": ([], []), "validation": ([], [])}
    B = hidden_batch.shape[0]
    hidden_cpu_batch = hidden_batch.detach().to("cpu")
    mask_cpu = attention_mask.detach().to("cpu")
    ids_cpu = input_ids.detach().to("cpu")

    for b in range(B):
        pid = prompt_ids_in_batch[b]
        rec = records[pid]
        split = split_assignments[pid]
        valid_positions = mask_cpu[b].nonzero(as_tuple=True)[0]
        token_rows, meta_rows = out[split]
        for token_position, pos in enumerate(valid_positions.tolist()):
            vec = hidden_cpu_batch[b, pos, :].clone()
            token_id = int(ids_cpu[b, pos].item())
            meta_rows.append({
                "prompt_id": pid, "domain": rec["domain"], "subdomain": rec["subdomain"],
                "source_model": rec.get("source_model"), "source_file": rec.get("source_file"),
                "token_position": token_position, "token_id": token_id,
                "selected_layer": selected_layer, "hook_location": hook_location,
                "activation_dtype": dtype_name, "split": split,
            })
            token_rows.append(vec)
    return out


def persist_shard(exp_dir, split, shard_record, meta_rows, manifest, metadata_jsonl_handle, row_counters):
    """Appends a freshly-flushed shard's metadata rows (with a global, monotonic
    activation_row_index per split) to the open metadata JSONL handle, and updates
    the in-memory manifest dict. Caller is responsible for writing manifest.json
    to disk after each shard (cheap -- it's a small JSON file) so progress is
    durable even if the process is killed mid-run."""
    start_row = row_counters[split]
    for i, meta in enumerate(meta_rows):
        meta["activation_row_index"] = start_row + i
        metadata_jsonl_handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
    row_counters[split] = start_row + len(meta_rows)
    metadata_jsonl_handle.flush()
    manifest.setdefault(split, {"shards": []})["shards"].append(shard_record)


def run_extraction(prompt_ids_to_process, records, split_assignments, exp_dir: Path,
                    batch_size: int, rows_per_shard: int, dtype_name: str = "bfloat16",
                    max_length: int = None, device: str = "cuda", log_every: int = 10,
                    data_root: Path = None):
    """
    Core extraction loop, shared by the pilot and full-run scripts. Processes
    exactly the prompt_ids in `prompt_ids_to_process` (callers are responsible
    for having already excluded completed prompts for resumption). Writes
    shards incrementally and returns a stats dict for the caller's report.

    `exp_dir` is always the main experiment directory (model_manifest.json is
    read from exp_dir/metadata/). `data_root` is where activations/ and the
    output metadata/ are actually written -- defaults to exp_dir, but the
    pilot script points it at an isolated subdirectory so pilot shards never
    mix with (or get mistaken for) the full-run's resumable shard manifest.
    """
    import time
    import torch

    if data_root is None:
        data_root = exp_dir
    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = data_root / "activations"

    device_name, gpu_name, total_vram_gb, free_vram_gb = mu.get_device_and_check_vram(min_free_gb=1.0)
    print(f"[extraction] GPU: {gpu_name}, free VRAM before load: {free_vram_gb:.2f} GB")

    model, tokenizer, revision = mu.load_model_and_tokenizer(dtype_name=dtype_name, device=device)
    # right-padding is required for correctness here: this is a single (non-generation)
    # forward pass over a causal model, so real tokens must occupy the sequence prefix;
    # with right-padding, every real token's activation is identical to the unpadded case
    # (causal attention never lets it attend to the trailing pad positions). Left-padding
    # would shift position_ids and corrupt RoPE for the real tokens -- asserted, not assumed.
    tokenizer.padding_side = "right"
    model_manifest = json.loads((exp_dir / "metadata" / "model_manifest.json").read_text(encoding="utf-8"))
    selected_layer = model_manifest["layer_selection"]["selected_layer_index"]
    hook_location = model_manifest["layer_selection"]["hook_location"]

    hook = ResidualStreamHook(model, selected_layer)

    _, manifest, _ = load_existing_progress(metadata_dir)
    row_counters = {
        split: sum(s["n_rows"] for s in manifest.get(split, {}).get("shards", []))
        for split in ("train", "validation")
    }

    writers = {
        "train": ShardWriter(activations_dir / "train", "train", rows_per_shard, manifest),
        "validation": ShardWriter(activations_dir / "validation", "validation", rows_per_shard, manifest),
    }

    meta_path = metadata_dir / "prompt_token_metadata.jsonl"
    meta_handle = meta_path.open("a", encoding="utf-8")

    batches, lengths = build_batches(
        {pid: records[pid] for pid in prompt_ids_to_process}, tokenizer, batch_size
    )

    torch.cuda.reset_peak_memory_stats()
    stats = {
        "n_prompts": 0, "n_tokens_retained": 0, "n_batches": 0,
        "failed_prompts": [], "batch_times_sec": [], "peak_vram_gb": 0.0,
        "truncated_prompts": [],
    }
    t_start = time.time()

    for batch_i, batch_pids in enumerate(batches):
        texts = [records[pid]["prompt"] for pid in batch_pids]
        tok_kwargs = {"padding": True, "return_tensors": "pt"}
        if max_length is not None:
            tok_kwargs.update(truncation=True, max_length=max_length)
        enc = tokenizer(texts, **tok_kwargs)
        for b_idx, pid in enumerate(batch_pids):
            orig_len = lengths[pid]
            if max_length is not None and orig_len > max_length:
                retained_len = int(enc["attention_mask"][b_idx].sum().item())
                stats["truncated_prompts"].append({"prompt_id": pid, "original_tokens": orig_len,
                                                     "retained_tokens": retained_len})
        enc = {k: v.to(device) for k, v in enc.items()}

        t0 = time.time()
        try:
            with torch.inference_mode():
                model(**enc, use_cache=False, output_hidden_states=False)
        except torch.cuda.OutOfMemoryError as e:
            stats["failed_prompts"].extend([{"prompt_id": pid, "reason": f"CUDA OOM: {e}"} for pid in batch_pids])
            torch.cuda.empty_cache()
            continue
        batch_time = time.time() - t0
        stats["batch_times_sec"].append(batch_time)

        split_outputs = extract_tokens_from_output(
            hook.captured, enc["attention_mask"], enc["input_ids"], batch_pids,
            records, None, split_assignments, selected_layer, hook_location, dtype_name,
        )
        for split, (token_rows, meta_rows) in split_outputs.items():
            if not token_rows:
                continue
            # add_prompt is called once per prompt (not once per batch) so a flush never
            # splits a single prompt's rows across shards; group this batch's rows by prompt.
            by_pid = {}
            for vec, meta in zip(token_rows, meta_rows):
                by_pid.setdefault(meta["prompt_id"], ([], []))
                by_pid[meta["prompt_id"]][0].append(vec)
                by_pid[meta["prompt_id"]][1].append(meta)
            for pid, (vecs, metas) in by_pid.items():
                writers[split].add_prompt(vecs, metas)
                for shard_record, shard_meta_rows in writers[split].written_shards:
                    persist_shard(exp_dir, split, shard_record, shard_meta_rows, manifest, meta_handle, row_counters)
                writers[split].written_shards = []
                (metadata_dir / "shard_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        stats["n_prompts"] += len(batch_pids)
        stats["n_tokens_retained"] += int(enc["attention_mask"].sum().item())
        stats["n_batches"] += 1
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        stats["peak_vram_gb"] = max(stats["peak_vram_gb"], peak_gb)

        del enc
        hook.captured = None
        torch.cuda.empty_cache()

        if batch_i % log_every == 0 or batch_i == len(batches) - 1:
            print(f"[extraction] batch {batch_i+1}/{len(batches)}, "
                  f"prompts_done={stats['n_prompts']}, tokens={stats['n_tokens_retained']}, "
                  f"peak_vram={stats['peak_vram_gb']:.2f}GB, last_batch_time={batch_time:.3f}s")

    # final flush of any partial shards
    for split, writer in writers.items():
        rec = writer.flush()
        if rec:
            persist_shard(exp_dir, split, rec, writer.written_shards[-1][1], manifest, meta_handle, row_counters)
            writer.written_shards = []
    (metadata_dir / "shard_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    meta_handle.close()
    hook.remove()

    stats["total_time_sec"] = time.time() - t_start
    stats["resolved_revision"] = revision
    return stats
