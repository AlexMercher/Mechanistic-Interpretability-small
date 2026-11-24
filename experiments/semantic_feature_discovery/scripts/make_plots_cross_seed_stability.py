"""
Phase 5, category D: cross_seed_stability. Six-way cross-seed reproducibility
classification counts (121-candidate denominator), cross-capacity classification
counts (61-candidate, >=medium-confidence denominator -- a DIFFERENT, smaller
population, per the reconciliation note), and the 12-combination threshold
sensitivity sweep. Reads only Phase 3 outputs.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "cross_seed_stability"
RECON_PATH = pc.DATA_DIR / "validation_results" / "cross_seed_capacity_reconciliation.json"
SWEEP_PATH = pc.DATA_DIR / "validation_results" / "cross_seed_sensitivity_sweep.json"

CLASSES = ["reproduced_strong", "numerically_matched_semantically_inconsistent",
           "possible_split", "possible_merge", "not_reproduced", "insufficient_evidence"]


def main():
    recon = pc.load_json(RECON_PATH)

    # D1: cross-seed six-way classification, seed_1 vs seed_2 (denominator = 121)
    csm = recon["cross_seed_matching"]
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(CLASSES))
    w = 0.35
    s1 = [csm["seed_1"][c] for c in CLASSES]
    s2 = [csm["seed_2"][c] for c in CLASSES]
    ax.bar(x - w / 2, s1, width=w, label=f"seed 1 (sum={csm['seed_1']['sum_check']})", color=pc.PALETTE[0])
    ax.bar(x + w / 2, s2, width=w, label=f"seed 2 (sum={csm['seed_2']['sum_check']})", color=pc.PALETTE[1])
    for i, (a, b) in enumerate(zip(s1, s2)):
        ax.text(i - w / 2, a + 0.5, str(a), ha="center", fontsize=10)
        ax.text(i + w / 2, b + 0.5, str(b), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=9)
    ax.set_ylabel("Number of candidates")
    ax.set_title(f"Cross-seed reproducibility, full {csm['denominator']}-candidate cohort\n"
                 "(full-target-dictionary search + full-source-dictionary reverse check, Confirmation-subset validated)")
    ax.legend()
    fig.text(0.5, -0.1,
              f"Caption: denominator = all 121 candidates, each matched against SAE-8192 "
              f"seed 1's and seed 2's COMPLETE 8192-feature dictionary. The same 25 feature "
              f"indices are insufficient_evidence under both seeds (identity-checked). "
              f"67/121 (55%) of candidates reproduce strongly under seed 2; this 67 is one "
              f"cell of the 121-row, not a separate smaller denominator.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "d1_cross_seed_classification",
                "Six-way cross-seed reproducibility classification counts for the full 121-candidate cohort, seed 1 vs. seed 2.",
                [RECON_PATH], n=csm["denominator"])

    # D2: cross-capacity classification, width 4096 vs 16384 (denominator = 61, DIFFERENT population)
    ccm = recon["cross_capacity_matching"]
    fig, ax = plt.subplots(figsize=(13, 7))
    w4 = [ccm["width_4096"][c] for c in CLASSES]
    w16 = [ccm["width_16384"][c] for c in CLASSES]
    ax.bar(x - w / 2, w4, width=w, label=f"width 4096 (sum={ccm['width_4096']['sum_check']})", color=pc.PALETTE[2])
    ax.bar(x + w / 2, w16, width=w, label=f"width 16384 (sum={ccm['width_16384']['sum_check']})", color=pc.PALETTE[3])
    for i, (a, b) in enumerate(zip(w4, w16)):
        ax.text(i - w / 2, a + 0.5, str(a), ha="center", fontsize=10)
        ax.text(i + w / 2, b + 0.5, str(b), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=9)
    ax.set_ylabel("Number of candidates")
    ax.set_title(f"Cross-capacity reproducibility, >=medium-confidence subset only ({ccm['denominator']}/121 candidates)")
    ax.legend()
    fig.text(0.5, -0.12,
              f"Caption: denominator = {ccm['denominator']} candidates at >=medium primary-review "
              "confidence (26 high + 35 medium) -- a DIFFERENT, smaller population than D1's "
              "full 121. insufficient_evidence is structurally 0 here because the >=medium "
              "filter already excludes thin-evidence candidates (not an independent finding).",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "d2_cross_capacity_classification",
                "Six-way cross-capacity reproducibility classification counts for the 61 >=medium-confidence candidates, width 4096 vs. 16384.",
                [RECON_PATH], n=ccm["denominator"])

    # D3: sensitivity sweep heatmap (seed_1), reproduced_strong count across cosine x margin grid
    sweep = pc.load_json(SWEEP_PATH)
    sw1 = [r for r in sweep if r["seed"] == "seed_1"]
    cosines = sorted(set(r["cosine_threshold"] for r in sw1))
    margins = sorted(set(r["margin_threshold"] for r in sw1))
    grid = np.full((len(margins), len(cosines)), np.nan)
    for r in sw1:
        i = margins.index(r["margin_threshold"])
        j = cosines.index(r["cosine_threshold"])
        grid[i, j] = r["classification_counts"].get("reproduced_strong", 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    for i in range(len(margins)):
        for j in range(len(cosines)):
            ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", color="white", fontsize=12)
    ax.set_xticks(range(len(cosines)))
    ax.set_xticklabels(cosines)
    ax.set_yticks(range(len(margins)))
    ax.set_yticklabels(margins)
    ax.set_xlabel("Decoder-cosine threshold")
    ax.set_ylabel("Runner-up margin threshold")
    ax.set_title("Threshold-sensitivity sweep: reproduced_strong count (seed 1, out of 121)\n"
                 "0.3 / 0.1 (the headline threshold) is a heuristic, not ground truth")
    fig.colorbar(im, ax=ax, label="reproduced_strong count")
    fig.text(0.5, -0.1,
              "Caption: 12 cosine x margin combinations (B.2 step 5); insufficient_evidence "
              "stays fixed at 25/121 across every combination since it is gated on evidence "
              "volume, not threshold -- confirming the sweep only moves the match/no-match "
              "boundary, not the evidence-floor outcome.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "d3_sensitivity_sweep_heatmap",
                "Cross-seed reproducibility threshold-sensitivity sweep: reproduced_strong count (seed 1) across 12 cosine x margin combinations.",
                [SWEEP_PATH], n=121)


if __name__ == "__main__":
    main()
