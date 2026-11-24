"""
Milestone 5, evaluation D: feature stability across the 3 independently
seeded training runs of a given width.

Feature indices are NOT aligned across independently trained SAEs (feature
37 in seed 0 has no a-priori relationship to feature 37 in seed 1) -- so
"stability" is measured via an explicit matching procedure, not index
correspondence, using two independent signals:

1. Decoder-direction matching: cosine similarity between unit-norm decoder
   columns (the learned dictionary directions in activation space).
2. Activation-pattern matching: cosine similarity between each feature's
   activation vector over a fixed sample of validation tokens (same tokens
   for both seeds in a pair) -- a functional/behavioral signal, independent
   of the decoder geometry.

Both use greedy one-to-one matching (highest-similarity pairs first,
skipping already-used indices) restricted to candidate pairs with cosine
similarity >= 0.3 (two random unit vectors in 2048-dim space have cosine
similarity with std ~= 1/sqrt(2048) =~ 0.022, so 0.3 is a strong signal,
not spurious). The cross-check between the two signals (do decoder-matched
pairs also have similar activation patterns?) is what distinguishes real
feature stability from mere decoder-geometry similarity.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_common as sc  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent.parent
MATCH_THRESHOLD = 0.3
ACTIVATION_SAMPLE_N = 8192


def summarize(arr: np.ndarray):
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()), "max": float(arr.max()),
    }


def greedy_match(sim: np.ndarray, threshold: float):
    idx_a, idx_b = np.where(sim >= threshold)
    vals = sim[idx_a, idx_b]
    order = np.argsort(-vals)
    used_a, used_b = set(), set()
    matches = []
    for k in order:
        i, j = int(idx_a[k]), int(idx_b[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, float(vals[k])))
    return matches


def pairwise_stability(width, seed_a, seed_b, device, preprocessor, val_x):
    model_a, _, _ = sc.load_model_for_eval(width, seed_a, "best_val", device)
    model_b, _, _ = sc.load_model_for_eval(width, seed_b, "best_val", device)

    with torch.no_grad():
        decoder_sim = (model_a.W_d.detach().t() @ model_b.W_d.detach()).cpu().numpy()

        gen = torch.Generator().manual_seed(1234)
        n = val_x.shape[0]
        sample_idx = torch.randperm(n, generator=gen)[:ACTIVATION_SAMPLE_N]
        raw = val_x[sample_idx].to(device, torch.float32)
        x = preprocessor.transform(raw)
        _, z_a = model_a(x)
        _, z_b = model_b(x)

    za = z_a.cpu().numpy()
    zb = z_b.cpu().numpy()
    za_norm = za / (np.linalg.norm(za, axis=0, keepdims=True) + 1e-8)
    zb_norm = zb / (np.linalg.norm(zb, axis=0, keepdims=True) + 1e-8)
    activation_sim = za_norm.T @ zb_norm

    decoder_matches = greedy_match(decoder_sim, MATCH_THRESHOLD)
    activation_matches = greedy_match(activation_sim, MATCH_THRESHOLD)

    cross_check_vals = np.array([activation_sim[i, j] for i, j, _ in decoder_matches])

    return {
        "decoder_n_matched": len(decoder_matches),
        "decoder_match_rate": len(decoder_matches) / width,
        "decoder_similarity_distribution": summarize(np.array([s for _, _, s in decoder_matches])),
        "activation_n_matched": len(activation_matches),
        "activation_match_rate": len(activation_matches) / width,
        "activation_similarity_distribution": summarize(np.array([s for _, _, s in activation_matches])),
        "decoder_matched_pairs_activation_similarity": summarize(cross_check_vals),
        "unmatched_decoder_fraction": 1.0 - len(decoder_matches) / width,
        "unmatched_activation_fraction": 1.0 - len(activation_matches) / width,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = sc.get_device_no_fallback()
    preprocessor = sc.Preprocessor.load(EXP_DIR / "preprocessing")
    val_x = sc.load_split_activations("validation")

    pair_results = {}
    for a, b in itertools.combinations(args.seeds, 2):
        key = f"seed{a}_vs_seed{b}"
        print(f"[evaluate_feature_stability] width={args.width} {key}")
        pair_results[key] = pairwise_stability(args.width, a, b, device, preprocessor, val_x)

    decoder_rates = [v["decoder_match_rate"] for v in pair_results.values()]
    activation_rates = [v["activation_match_rate"] for v in pair_results.values()]

    out = {
        "dictionary_width": args.width,
        "seeds_compared": args.seeds,
        "match_threshold": MATCH_THRESHOLD,
        "activation_sample_n": ACTIVATION_SAMPLE_N,
        "pairwise": pair_results,
        "summary": {
            "mean_decoder_match_rate": float(np.mean(decoder_rates)),
            "mean_activation_match_rate": float(np.mean(activation_rates)),
        },
        "note": "Feature indices are NOT aligned across independently trained SAEs -- "
                "matches above come from explicit greedy cosine-similarity matching, "
                "not index correspondence.",
    }

    out_path = EXP_DIR / "evaluations" / "feature_stability" / f"sae_{args.width}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[evaluate_feature_stability] wrote {out_path}")


if __name__ == "__main__":
    main()
