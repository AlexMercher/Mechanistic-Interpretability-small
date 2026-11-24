"""
Phase 5, category A: feature_distributions. Population-level activation
frequency structure of the full SAE-8192 (seed 0) dictionary on the Discovery
subset, plus where the 121 selected candidates sit within that population.
Reads only Phase 1 outputs; computes no new activations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "feature_distributions"
INV_PATH = pc.DATA_DIR / "feature_analysis" / "sae_8192_seed0_inventory_discovery.jsonl"
CAND_PATH = pc.DATA_DIR / "feature_analysis" / "candidates.jsonl"


def main():
    inv = pd.DataFrame(pc.load_jsonl(INV_PATH))
    cand = pd.DataFrame(pc.load_jsonl(CAND_PATH))
    n_dead = int(inv["is_dead_in_discovery"].sum())
    n_alive = len(inv) - n_dead

    # A1: token-level firing frequency histogram, alive features only, log-x
    fig, ax = plt.subplots(figsize=(9, 6))
    alive = inv[~inv["is_dead_in_discovery"]]
    ax.hist(alive["token_level_frequency"].clip(lower=1e-6), bins=60, log=True,
            color=pc.PALETTE[0], edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Token-level firing frequency (Discovery subset, log scale)")
    ax.set_ylabel("Number of SAE-8192 features (log scale)")
    ax.set_title(f"SAE-8192 seed 0 feature activation frequency, Discovery subset\n"
                 f"(n={len(inv)} total features; {n_dead} dead / {n_alive} alive)")
    fig.text(0.5, -0.08,
              f"Caption: distribution of per-feature token firing frequency among the "
              f"{n_alive} features that fire at least once in the 1,265-prompt Discovery "
              f"subset (49,615 tokens). {n_dead} of 8,192 features never fire and are "
              f"excluded from this histogram (reported separately, not hidden).",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "a1_token_frequency_histogram",
                "Token-level firing-frequency histogram across all alive SAE-8192 features (Discovery subset).",
                [INV_PATH], n=len(inv))

    # A2: primary_stratum counts among the 121 candidates
    fig, ax = plt.subplots(figsize=(11, 6))
    import seaborn as sns
    order = cand["primary_stratum"].value_counts().index.tolist()
    sns.countplot(data=cand, y="primary_stratum", order=order, ax=ax, color=pc.PALETTE[1])
    for i, stratum in enumerate(order):
        c = int((cand["primary_stratum"] == stratum).sum())
        ax.text(c + 0.3, i, str(c), va="center", fontsize=11)
    ax.set_xlabel("Number of candidates (primary stratum)")
    ax.set_ylabel("Selection stratum")
    ax.set_title(f"121-candidate cohort by primary selection stratum (B.1 quota)")
    fig.text(0.5, -0.05,
              "Caption: each of the 121 frozen candidates is counted once, under the first "
              "B.1 quota row it matched (primary stratum); secondary-stratum memberships are "
              "tracked separately and not double-counted here.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "a2_candidate_strata_counts",
                "Count of the 121 selected candidates by primary selection stratum.",
                [CAND_PATH], n=len(cand))

    # A3: token freq vs prompt freq scatter, colored by stratum
    fig, ax = plt.subplots(figsize=(10, 7))
    strata = cand["primary_stratum"].unique()
    for i, s in enumerate(strata):
        sub = cand[cand["primary_stratum"] == s]
        ax.scatter(sub["token_level_frequency"], sub["prompt_level_frequency"],
                   label=s, color=pc.PALETTE[i % len(pc.PALETTE)], alpha=0.75, s=50)
    ax.set_xscale("log")
    ax.set_xlabel("Token-level frequency (Discovery subset, log scale)")
    ax.set_ylabel("Prompt-level frequency (Discovery subset)")
    ax.set_title("Candidate token-level vs. prompt-level firing frequency, by selection stratum")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.5, -0.06,
              f"Caption: n={len(cand)} candidates. Token- and prompt-level frequency are "
              "both reported per B.1; they diverge for features that fire many times within "
              "few prompts vs. once across many prompts.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "a3_token_vs_prompt_frequency_scatter",
                "Token-level vs. prompt-level firing frequency for the 121 candidates, colored by stratum.",
                [CAND_PATH], n=len(cand))


if __name__ == "__main__":
    main()
