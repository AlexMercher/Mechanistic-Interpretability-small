# SAE Capacity Comparison (4096 / 8192 / 16384)

| Metric | 4096 | 8192 | 16384 |
|---|---:|---:|---:|
| Validation reconstruction MSE (mean, 3 seeds) | 1.898 | 1.766 | 1.788 |
|   ...range across seeds | [1.888, 1.906] | [1.764, 1.769] | [1.781, 1.8] |
| Explained variance (mean, 3 seeds, saturated ~0.999) | 0.9991 | 0.9992 | 0.9992 |
| Mean active features/token (mean, 3 seeds) | 44.2 | 40.17 | 42.73 |
|   ...range across seeds | [43.66, 44.61] | [39.99, 40.45] | [42.36, 43.03] |
| Median active features/token (mean, 3 seeds) | 38.33 | 32 | 31.33 |
| Dead features (mean count, 3 seeds) | 1898 | 3623 | 8938 |
| Dead feature % (mean, 3 seeds) | 46.33 | 44.23 | 54.56 |
|   ...range across seeds | [44.78, 47.75] | [43.15, 45.19] | [54.12, 55.24] |
| Rare feature % (mean, 3 seeds) | 0.3499 | 0.2035 | 0.1302 |
| Active feature % (mean, 3 seeds) | 53.67 | 55.77 | 45.44 |
| Feature stability, raw (decoder match / full width) | 0.4233 | 0.3138 | 0.1565 |
| Feature stability, live-normalized (decoder match / live fraction) | 0.7888 | 0.5626 | 0.3443 |
| EV per active feature (~1/L0, NOT independent of sparsity -- see note) | 0.0226 | 0.02488 | 0.02338 |
| Training duration (sec, avg over seeds) | 453.9 | 998.2 | 1315 |
| Peak VRAM (GB, avg over seeds) | 0.76 | 1.29 | 2.37 |
| Checkpoint size (bytes) | 201407742 | 402783486 | 805534974 |

## Verdict notes

- Lowest validation reconstruction MSE (mean over 3 seeds): 8192 (spread across widths is only 7.5% -- see per-width ranges in the table; reconstruction barely differentiates the three widths here).
- 'Explained-variance-per-active-feature' is NOT independent corroboration of the reconstruction result: explained variance is saturated (~0.999) at every width, so this ratio is numerically almost exactly 1/mean_L0 -- it just restates the sparsity ranking, not a genuine recon-vs-sparsity trade-off being navigated in this result.
- Lowest dead-feature percentage (mean over 3 seeds): 8192.
- Highest cross-seed decoder-match stability, raw (divided by full dictionary width): 4096. Live-normalized (divided by the live/non-dead feature fraction instead, which removes the double penalty wide dictionaries take from also having more dead features): 4096.
- No single width wins on every axis -- reconstruction (barely differentiated), dead-feature rate, and stability do not all point the same way. This script deliberately does not collapse these into one score or declare the widest dictionary the winner by default. Reconstruction/sparsity/dead-feature figures above are now 3-seed aggregates (not single-seed point estimates); see reports/opus_final_review.md for the full methodology review.

Reconstruction (validation MSE / explained variance) and sparsity (mean L0, L1) are reported as co-equal primary axes, not ranked with reconstruction as primary and sparsity as a tiebreaker. 'explained_variance_per_active_feature' is one additional joint lens (reconstruction quality achieved per unit of sparsity spent), not a replacement for looking at both axes independently.
