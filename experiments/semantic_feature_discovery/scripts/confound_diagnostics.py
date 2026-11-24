"""
Milestone 6, Phase 2 step 1 (plan D.2.1): template (tiered)/domain-normalized/script
confound diagnostics, computed on the Discovery and Confirmation subsets SEPARATELY
and compared side by side.

This doubles as a robustness check (plan B.3): a genuine pattern should look similar
on both subsets; a Discovery-subset artifact may not replicate on Confirmation, which
is itself a useful, reportable finding -- flagged explicitly here, not smoothed over.

Does not modify select_candidates.py's output or the 121-candidate cohort in any way.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
DIVERGENCE_RELATIVE_THRESHOLD = 0.5  # >50% relative change between subsets flagged


def subset_domain_language_baseline(subset_label: str, records: dict) -> dict:
    """Computes this subset's OWN domain and language_form token-share baselines
    directly from prompt_token_metadata.jsonl (not reusing the other subset's or any
    prior-milestone baseline)."""
    assignment = {}
    with (SFD_DIR / "configs" / "discovery_confirmation_assignment.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assignment[row["prompt_id"]] = row["subset"]
    lang_annot = sf.load_language_script_annotations()

    domains, langs = [], []
    with sf.PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != "validation" or assignment.get(row["prompt_id"]) != subset_label:
                continue
            domains.append(row["domain"])
            langs.append(lang_annot.get(row["prompt_id"], {}).get("language_form", "unknown"))

    n = len(domains)
    domain_share = pd.Series(domains).value_counts().div(n).to_dict()
    lang_share = pd.Series(langs).value_counts().div(n).to_dict()
    return {"n_tokens": n, "domain_token_share": domain_share, "language_form_token_share": lang_share}


def per_candidate_metrics(events: pd.DataFrame, feat: int, domain_baseline: dict, lang_baseline: dict) -> dict:
    feat_events = events[events["feature_idx"] == feat]
    total = len(feat_events)
    if total == 0:
        return {"n_firing_tokens": 0, "domain_concentration": None, "cross_domain_entropy_norm": None,
                "script_language_score": None, "domain_firing_counts": {}}

    domain_counts = feat_events["domain"].value_counts().to_dict()
    domain_concentration = max(
        (c / total) / domain_baseline.get(d, 1e-9) for d, c in domain_counts.items())
    entropy = 0.0
    for d, c in domain_counts.items():
        p = c / total
        if p > 0:
            entropy -= p * np.log2(p)
    entropy_norm = entropy / np.log2(5)

    lang_counts = feat_events["language_form"].value_counts().to_dict() if "language_form" in feat_events.columns else {}
    script_score = 0.0
    for form in ("native_hindi", "romanized_hindi"):
        c = lang_counts.get(form, 0)
        p = c / total
        b = lang_baseline.get(form, 1e-9)
        script_score = max(script_score, p / b)

    return {"n_firing_tokens": total, "domain_concentration": domain_concentration,
            "cross_domain_entropy_norm": entropy_norm, "script_language_score": script_score,
            "domain_firing_counts": domain_counts}


def flag_divergence(discovery_val, confirmation_val):
    if discovery_val is None or confirmation_val is None:
        return "insufficient_confirmation_evidence" if confirmation_val is None else "insufficient_discovery_evidence"
    if discovery_val == 0 and confirmation_val == 0:
        return "consistent"
    denom = max(abs(discovery_val), 1e-9)
    rel_change = abs(confirmation_val - discovery_val) / denom
    return "large_divergence" if rel_change > DIVERGENCE_RELATIVE_THRESHOLD else "consistent"


def main():
    candidates = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8")]
    records = sf.load_normalized_records()
    lang_annot = sf.load_language_script_annotations()

    events = {}
    baselines = {}
    for subset in ("discovery", "confirmation"):
        ev = pd.read_pickle(SFD_DIR / "analysis" / "data" / "feature_analysis" / f"candidate_firing_events_{subset}.pkl")
        ev["language_form"] = ev["prompt_id"].map(lambda pid: lang_annot.get(pid, {}).get("language_form", "unknown"))
        events[subset] = ev
        baselines[subset] = subset_domain_language_baseline(subset, records)
        print(f"[confound] {subset}: baseline from {baselines[subset]['n_tokens']} tokens")

    evidence = {}
    for subset in ("discovery", "confirmation"):
        evidence[subset] = {}
        with (SFD_DIR / "analysis" / "data" / "feature_analysis" / f"candidate_evidence_{subset}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                evidence[subset][row["feature_idx"]] = row

    out_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / "confound_diagnostics.jsonl"
    n_large_divergence = 0
    with out_path.open("w", encoding="utf-8") as f:
        for cand in candidates:
            feat = cand["feature_idx"]
            per_subset = {}
            for subset in ("discovery", "confirmation"):
                m = per_candidate_metrics(events[subset], feat, baselines[subset]["domain_token_share"],
                                           baselines[subset]["language_form_token_share"])
                m["template_diversity"] = evidence[subset][feat]["template_diversity"]
                m["subdomain_diversity"] = evidence[subset][feat]["subdomain_diversity"]
                m["lexical_diversity"] = evidence[subset][feat]["lexical_diversity"]
                per_subset[subset] = m

            divergence = {
                "domain_concentration": flag_divergence(per_subset["discovery"]["domain_concentration"], per_subset["confirmation"]["domain_concentration"]),
                "cross_domain_entropy_norm": flag_divergence(per_subset["discovery"]["cross_domain_entropy_norm"], per_subset["confirmation"]["cross_domain_entropy_norm"]),
                "script_language_score": flag_divergence(per_subset["discovery"]["script_language_score"], per_subset["confirmation"]["script_language_score"]),
            }
            if "large_divergence" in divergence.values():
                n_large_divergence += 1

            f.write(json.dumps({
                "feature_idx": feat, "primary_stratum": cand["primary_stratum"],
                "discovery": per_subset["discovery"], "confirmation": per_subset["confirmation"],
                "divergence_flags": divergence,
            }) + "\n")

    print(f"[confound] wrote {out_path}")
    print(f"[confound] {n_large_divergence}/{len(candidates)} candidates show >50% relative "
          f"divergence between Discovery and Confirmation on at least one confound metric "
          f"(expected for some candidates -- reported, not corrected)")


if __name__ == "__main__":
    main()
