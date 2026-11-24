#!/usr/bin/env python
"""
Milestone 4, step 1 -- inspect Qwen3-1.7B's architecture, tokenizer, and
GPU/VRAM situation BEFORE committing to an extraction pipeline design.
Downloads the checkpoint into weights/hf_cache/ if not already present
(reused on every subsequent run -- never re-downloaded).

Writes metadata/model_manifest.json with everything the rest of this
experiment needs to hard-code safely: exact revision, hidden_size,
num_layers, chosen middle layer index + hook location, special tokens.

Usage:
    python inspect_model.py [--dtype bfloat16]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_utils as mu  # noqa: E402 -- must be imported before torch/transformers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent.parent / "metadata")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    print("[inspect_model] checking GPU/VRAM ...")
    device, gpu_name, total_vram_gb, free_vram_gb = mu.get_device_and_check_vram(min_free_gb=2.0)
    print(f"[inspect_model] GPU: {gpu_name}, total VRAM={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB")

    print(f"[inspect_model] loading {mu.MODEL_NAME} (cache={mu.WEIGHTS_CACHE_DIR}, dtype={args.dtype}) ...")
    model, tokenizer, revision = mu.load_model_and_tokenizer(dtype_name=args.dtype, device=device)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    # Qwen3 decoder layers live at model.model.layers[i]; the residual stream AFTER
    # layer i is the output of model.model.layers[i] (a tuple, first element is the
    # hidden state) -- i.e. hook the module's forward output, not its input.
    middle_layer_zero_based = n_layers // 2  # architectural midpoint, zero-based index
    module_name = f"model.layers[{middle_layer_zero_based}]"

    after_alloc_free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    vram_after_load_gb = (total_bytes - after_alloc_free_bytes) / 1e9

    special_tokens = {
        "bos_token": tokenizer.bos_token, "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token, "unk_token_id": tokenizer.unk_token_id,
    }
    if tokenizer.pad_token_id is None:
        print("[inspect_model] WARNING: tokenizer has no pad_token -- will set pad_token=eos_token "
              "for batched inference (documented, not silent).")

    # sanity forward pass
    sample_text = "query: What is the mechanism of DNA replication?"
    enc = tokenizer(sample_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    # output_hidden_states gives a tuple of len n_layers+1: [0]=embedding output, [i]=output of layer i-1 (1-based into the tuple)
    n_hidden_states = len(out.hidden_states)
    sample_shape = list(out.hidden_states[middle_layer_zero_based + 1].shape)  # +1 because index 0 is embeddings

    manifest = {
        "model_name": mu.MODEL_NAME,
        "resolved_revision": revision,
        "loading_precision": args.dtype,
        "activation_storage_precision": "bfloat16",  # chosen to match loading precision; see model_selection notes
        "architecture": {
            "num_hidden_layers": n_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
            "attn_implementation": model.config._attn_implementation,
        },
        "layer_selection": {
            "total_layers": n_layers,
            "indexing": "zero_based",
            "selected_layer_index": middle_layer_zero_based,
            "selection_rule": "n_layers // 2 (architectural midpoint, zero-based)",
            "module_name": module_name,
            "hook_location": "output of model.model.layers[i] (the decoder block), i.e. the "
                              "residual stream AFTER that Transformer block has been applied -- "
                              "NOT attention output, NOT MLP output alone, NOT the embedding layer, "
                              "NOT logits.",
            "equivalent_output_hidden_states_index": middle_layer_zero_based + 1,
            "note": "output_hidden_states[0] is the embedding layer's output (residual stream "
                    "before any Transformer block); output_hidden_states[k] for k>=1 is the "
                    "residual stream after decoder block (k-1). This experiment extracts "
                    "output_hidden_states[selected_layer_index + 1], i.e. after block "
                    "selected_layer_index (zero-based).",
        },
        "special_tokens": special_tokens,
        "sample_forward_pass": {
            "sample_text": sample_text,
            "n_hidden_states_returned": n_hidden_states,
            "expected_n_hidden_states": n_layers + 1,
            "middle_layer_activation_shape": sample_shape,
            "hidden_size_matches_config": sample_shape[-1] == hidden_size,
        },
        "gpu": {
            "device_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 3),
            "free_vram_gb_before_load": round(free_vram_gb, 3),
            "vram_used_after_load_and_sample_forward_gb": round(vram_after_load_gb, 3),
        },
        "weights_cache_dir": str(mu.WEIGHTS_CACHE_DIR),
    }

    out_path = args.out_dir / "model_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[inspect_model] wrote {out_path}")
    print(json.dumps(manifest, indent=2))

    assert sample_shape[-1] == hidden_size, "activation dim does not match hidden_size!"
    assert n_hidden_states == n_layers + 1, "unexpected number of hidden states!"
    print("[inspect_model] all sanity checks passed.")


if __name__ == "__main__":
    main()
