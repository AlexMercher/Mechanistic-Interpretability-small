"""
Milestone 5, final step: build the capacity-comparison table and narrative
verdict from the evaluation JSONs already written by evaluate_reconstruction.py,
evaluate_sparsity.py, evaluate_feature_utilization.py, evaluate_feature_stability.py
-- never hand-typed.

Explicit framing (per the task and the research-point clarification given
during planning): reconstruction quality and sparsity are treated as
CO-EQUAL primary axes, not reconstruction-first-with-sparsity-as-tiebreaker.
The comparison therefore reports both independently, adds a joint
"reconstruction-per-active-feature" efficiency ratio as one additional lens
(not a single blended score that would hide the trade-off), and the verdict
section explicitly checks whether a width that "wins" on one axis holds up
on the other before calling it better. The largest dictionary is never
declared the winner merely for lower raw reconstruction error.
"""
import json
import re
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
WIDTHS = [4096, 8192, 16384]


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


LOG_FILES = ["plateau_run_8192_seed0.log", "seed0_runs.log", "seeds_1_2_runs.log"]


def parse_training_timings():
    """Scans every known training log for per-(width,seed) total_time_sec and
    peak_vram_gb, printed by train_sae.py's final summary line. Tracks the
    current (width, seed) from each run's startup line rather than relying on
    a specific driver script's "=== START ===" markers, since these runs were
    launched from several different ad-hoc driver scripts/foreground calls."""
    timings = {}
    current = None
    device_re = re.compile(r"\[train_sae\] device=\S+ width=(\d+) seed=(\d+)")
    done_re = re.compile(r"\[train_sae\] done\. steps=(\d+) total_time_sec=([\d.]+) peak_vram_gb=([\d.]+)")
    for fname in LOG_FILES:
        log_path = EXP_DIR / "logs" / fname
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = device_re.search(line)
            if m:
                current = (int(m.group(1)), int(m.group(2)))
                continue
            m = done_re.search(line)
            if m and current:
                width, seed = current
                timings.setdefault(width, {})[seed] = {
                    "steps": int(m.group(1)), "total_time_sec": float(m.group(2)),
                    "peak_vram_gb": float(m.group(3)),
                }
                current = None
    return timings


def checkpoint_size_bytes(width, seed=0):
    path = EXP_DIR / "checkpoints" / f"sae_{width}" / f"seed_{seed}" / "final.pt"
    if path.exists():
        return path.stat().st_size
    return None


SEEDS = [0, 1, 2]


def agg(values):
    """Returns (mean, min, max) for a list of per-seed scalar values -- used
    so every headline metric is an across-seed aggregate, not a single-seed
    point estimate (seed 0 alone understated how much seeds 0-2 actually
    diverge, per the Opus review pass -- see reports/opus_final_review.md)."""
    return sum(values) / len(values), min(values), max(values)


def main():
    stability = {w: load_json(EXP_DIR / "evaluations" / "feature_stability" / f"sae_{w}.json") for w in WIDTHS}
    timings = parse_training_timings()

    table = {}
    for w in WIDTHS:
        recon_seeds, sparsity_seeds, util_seeds = [], [], []
        for seed in SEEDS:
            r = load_json(EXP_DIR / "evaluations" / "reconstruction" / f"sae_{w}_seed{seed}.json")
            s = load_json(EXP_DIR / "evaluations" / "sparsity" / f"sae_{w}_seed{seed}.json")
            u = load_json(EXP_DIR / "evaluations" / "feature_utilization" / f"sae_{w}_seed{seed}.json")
            if r is None or s is None or u is None:
                continue
            recon_seeds.append(r)
            sparsity_seeds.append(s)
            util_seeds.append(u)

        if len(recon_seeds) < len(SEEDS):
            table[w] = {"status": f"incomplete -- only {len(recon_seeds)}/{len(SEEDS)} seeds evaluated"}
            continue

        st = stability.get(w)
        widths_timings = timings.get(w, {})
        avg_time = (sum(t["total_time_sec"] for t in widths_timings.values()) / len(widths_timings)
                    if widths_timings else None)
        avg_vram = (sum(t["peak_vram_gb"] for t in widths_timings.values()) / len(widths_timings)
                    if widths_timings else None)
        ckpt_size = checkpoint_size_bytes(w)

        mse_mean, mse_min, mse_max = agg([r["mean_squared_reconstruction_error_per_vector"] for r in recon_seeds])
        ev_mean, ev_min, ev_max = agg([r["explained_variance_overall"] for r in recon_seeds])
        l0_mean, l0_min, l0_max = agg([s["mean_active_features_per_token"] for s in sparsity_seeds])
        med_l0_mean, _, _ = agg([s["median_active_features_per_token"] for s in sparsity_seeds])
        l1_mean, _, _ = agg([s["mean_l1_activation_magnitude"] for s in sparsity_seeds])
        dead_mean, dead_min, dead_max = agg([u["dead_features_pct"] for u in util_seeds])
        rare_mean, _, _ = agg([u["rare_features_pct"] for u in util_seeds])
        active_mean, active_min, active_max = agg([u["features_active_at_least_once_pct"] for u in util_seeds])
        dead_counts_mean, _, _ = agg([u["dead_features"] for u in util_seeds])

        # Explained variance is saturated (~0.999) at every width (see opus_final_review.md
        # Issue 2), so EV/L0 is numerically ~= 1/L0 here -- it is NOT an independent
        # reconstruction-sparsity trade-off signal in this particular result, just a
        # restatement of the L0 ranking. Kept for transparency, captioned accordingly.
        efficiency_ratio = ev_mean / l0_mean if l0_mean else None

        decoder_match_mean = st["summary"]["mean_decoder_match_rate"] if st else None
        # Raw decoder_match_rate divides matched-feature count by the FULL dictionary width,
        # so widths with more dead features are penalized twice (once in dead_feature_pct,
        # again here, since dead features can't match across seeds). Live-normalized rate
        # divides by the live (non-dead) feature fraction instead -- see opus_final_review.md
        # Issue 3. Both are reported; neither alone is "the" stability number.
        live_fraction = active_mean / 100.0 if active_mean else None
        decoder_match_live_normalized = (decoder_match_mean / live_fraction
                                          if decoder_match_mean is not None and live_fraction else None)

        table[w] = {
            "validation_reconstruction_mse_mean": mse_mean,
            "validation_reconstruction_mse_range": [mse_min, mse_max],
            "explained_variance_mean": ev_mean,
            "explained_variance_range": [ev_min, ev_max],
            "mean_active_features_per_token_mean": l0_mean,
            "mean_active_features_per_token_range": [l0_min, l0_max],
            "median_active_features_per_token_mean": med_l0_mean,
            "mean_l1_activation_magnitude_mean": l1_mean,
            "dead_features_mean": dead_counts_mean,
            "dead_feature_pct_mean": dead_mean,
            "dead_feature_pct_range": [dead_min, dead_max],
            "rare_feature_pct_mean": rare_mean,
            "active_feature_pct_mean": active_mean,
            "active_feature_pct_range": [active_min, active_max],
            "feature_stability_mean_decoder_match_rate": decoder_match_mean,
            "feature_stability_decoder_match_rate_live_normalized": decoder_match_live_normalized,
            "feature_stability_mean_activation_match_rate": st["summary"]["mean_activation_match_rate"] if st else None,
            "explained_variance_per_active_feature": efficiency_ratio,
            "avg_training_duration_sec": avg_time,
            "avg_peak_vram_gb": avg_vram,
            "checkpoint_size_bytes": ckpt_size,
            "n_seeds_aggregated": len(recon_seeds),
        }

    complete_widths = [w for w in WIDTHS if "status" not in table[w]]

    verdict_lines = []
    if len(complete_widths) == len(WIDTHS):
        best_recon = min(complete_widths, key=lambda w: table[w]["validation_reconstruction_mse_mean"])
        least_dead_pct = min(complete_widths, key=lambda w: table[w]["dead_feature_pct_mean"])
        most_stable_raw = max(
            [w for w in complete_widths if table[w]["feature_stability_mean_decoder_match_rate"] is not None],
            key=lambda w: table[w]["feature_stability_mean_decoder_match_rate"],
            default=None,
        )
        most_stable_live = max(
            [w for w in complete_widths if table[w]["feature_stability_decoder_match_rate_live_normalized"] is not None],
            key=lambda w: table[w]["feature_stability_decoder_match_rate_live_normalized"],
            default=None,
        )
        mse_vals = {w: table[w]["validation_reconstruction_mse_mean"] for w in complete_widths}
        mse_spread_pct = 100 * (max(mse_vals.values()) - min(mse_vals.values())) / min(mse_vals.values())

        verdict_lines.append(f"Lowest validation reconstruction MSE (mean over {len(SEEDS)} seeds): {best_recon} "
                              f"(spread across widths is only {mse_spread_pct:.1f}% -- see per-width ranges in the table; "
                              f"reconstruction barely differentiates the three widths here).")
        verdict_lines.append(
            "'Explained-variance-per-active-feature' is NOT independent corroboration of the reconstruction "
            "result: explained variance is saturated (~0.999) at every width, so this ratio is numerically "
            "almost exactly 1/mean_L0 -- it just restates the sparsity ranking, not a genuine recon-vs-sparsity "
            "trade-off being navigated in this result."
        )
        verdict_lines.append(f"Lowest dead-feature percentage (mean over {len(SEEDS)} seeds): {least_dead_pct}.")
        if most_stable_raw is not None:
            verdict_lines.append(
                f"Highest cross-seed decoder-match stability, raw (divided by full dictionary width): {most_stable_raw}. "
                f"Live-normalized (divided by the live/non-dead feature fraction instead, which removes the double "
                f"penalty wide dictionaries take from also having more dead features): {most_stable_live}."
            )
        axis_winners = {best_recon, least_dead_pct}
        if most_stable_raw is not None:
            axis_winners.add(most_stable_raw)
        if len(axis_winners) == 1:
            verdict_lines.append(f"{best_recon} wins on every axis -- a clean, non-confounded recommendation.")
        else:
            verdict_lines.append(
                "No single width wins on every axis -- reconstruction (barely differentiated), dead-feature "
                "rate, and stability do not all point the same way. This script deliberately does not collapse "
                "these into one score or declare the widest dictionary the winner by default. Reconstruction/"
                "sparsity/dead-feature figures above are now 3-seed aggregates (not single-seed point estimates); "
                "see reports/opus_final_review.md for the full methodology review."
            )

    out = {
        "table": table,
        "widths_compared": WIDTHS,
        "complete_widths": complete_widths,
        "verdict_notes": verdict_lines,
        "methodology_note": (
            "Reconstruction (validation MSE / explained variance) and sparsity (mean L0, L1) "
            "are reported as co-equal primary axes, not ranked with reconstruction as primary "
            "and sparsity as a tiebreaker. 'explained_variance_per_active_feature' is one "
            "additional joint lens (reconstruction quality achieved per unit of sparsity spent), "
            "not a replacement for looking at both axes independently."
        ),
    }

    (EXP_DIR / "reports" / "capacity_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md_lines = ["# SAE Capacity Comparison (4096 / 8192 / 16384)\n"]
    md_lines.append("| Metric | 4096 | 8192 | 16384 |")
    md_lines.append("|---|---:|---:|---:|")
    metric_order = [
        ("validation_reconstruction_mse_mean", "Validation reconstruction MSE (mean, 3 seeds)"),
        ("validation_reconstruction_mse_range", "  ...range across seeds"),
        ("explained_variance_mean", "Explained variance (mean, 3 seeds, saturated ~0.999)"),
        ("mean_active_features_per_token_mean", "Mean active features/token (mean, 3 seeds)"),
        ("mean_active_features_per_token_range", "  ...range across seeds"),
        ("median_active_features_per_token_mean", "Median active features/token (mean, 3 seeds)"),
        ("dead_features_mean", "Dead features (mean count, 3 seeds)"),
        ("dead_feature_pct_mean", "Dead feature % (mean, 3 seeds)"),
        ("dead_feature_pct_range", "  ...range across seeds"),
        ("rare_feature_pct_mean", "Rare feature % (mean, 3 seeds)"),
        ("active_feature_pct_mean", "Active feature % (mean, 3 seeds)"),
        ("feature_stability_mean_decoder_match_rate", "Feature stability, raw (decoder match / full width)"),
        ("feature_stability_decoder_match_rate_live_normalized", "Feature stability, live-normalized (decoder match / live fraction)"),
        ("explained_variance_per_active_feature", "EV per active feature (~1/L0, NOT independent of sparsity -- see note)"),
        ("avg_training_duration_sec", "Training duration (sec, avg over seeds)"),
        ("avg_peak_vram_gb", "Peak VRAM (GB, avg over seeds)"),
        ("checkpoint_size_bytes", "Checkpoint size (bytes)"),
    ]
    for key, label in metric_order:
        row = [label]
        for w in WIDTHS:
            v = table.get(w, {}).get(key)
            if v is None:
                row.append("n/a")
            elif isinstance(v, list):
                row.append(f"[{v[0]:.4g}, {v[1]:.4g}]")
            elif isinstance(v, float):
                row.append(f"{v:.4g}")
            else:
                row.append(str(v))
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("\n## Verdict notes\n")
    for line in verdict_lines:
        md_lines.append(f"- {line}")

    md_lines.append(f"\n{out['methodology_note']}\n")

    (EXP_DIR / "reports" / "capacity_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("[compare_sae_capacities] wrote reports/capacity_comparison.json and .md")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
