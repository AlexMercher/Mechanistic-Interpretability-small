# Milestone 6 Candidate Selection Report

Total candidates selected: 121

## Primary-stratum counts

rare_but_active                                   20
high_frequency                                    15
medium_frequency                                  15
strong_domain_concentration                       15
cross_domain_broadly_distributed                  15
suspicious_template_concentration                 10
script_language_dependence                        10
cross_seed_prescreened_reproducible               10
cross_seed_prescreened_nonreproducible_control    10
always_on_near_ubiquitous                          1


## Discovery/Confirmation boundary (mandatory disclosure)

The two cross-seed strata (`cross_seed_prescreened_reproducible`, `cross_seed_prescreened_nonreproducible_control`) were selected using a **Discovery-subset-only** preliminary decoder+activation-pattern screen (0.3 cosine / 0.1 margin thresholds) against the complete SAE-8192 seed-1 and seed-2 dictionaries. This preliminary result decided ONLY which features enter these two strata -- it is never reused as, or averaged into, the final cross-seed validation number. The independent, final cross-seed activation-pattern match, top-prompt overlap, and matched-target semantic evidence pull (for every selected candidate, not just these two strata) are computed fresh in Phase 3 using **Confirmation-subset-only** activations.

## Rare-feature coverage limitation (mandatory disclosure)

Candidate selection runs entirely on the Discovery-subset inventory. A feature that never fires in Discovery but does fire in Confirmation cannot enter the `rare_but_active` stratum, or any stratum. This cohort is a stratified sample of Discovery-visible features, not an exhaustive census of all rare features in the dictionary.

**Measured result (Phase 2, `build_confirmation_only_coverage.py`):** of the 3,535 features dead in Discovery, **0 fire in Confirmation** (`confirmation_only_feature_coverage.json`). Every Discovery-dead feature is also dead in Confirmation for this SAE-8192 seed-0 checkpoint on this validation split -- the coverage gap this limitation describes exists in principle (and would matter for a sparser split or a wider dictionary) but did not affect any feature in this specific run. This is a measured outcome, not evidence the limitation is moot in general.
