"""
Milestone 6, Phase 3 pre-step: merges every Phase 1/2 output into one consolidated
per-candidate evidence bundle, so reviewers (primary and the blind second reviewer)
receive the full evidence set the task requires -- never just a feature index or a
handful of top prompts.

Each bundle contains: selection metadata (primary/secondary strata, selection metrics),
Discovery examples (top/mid/low/one-per-template) used to form the initial hypothesis,
Confirmation examples used to test it, confound diagnostics on both subsets side by
side with divergence flags, and counterexamples (Confirmation-subset only). Does not
interpret anything -- purely a merge/reshape step.
"""
import json
from pathlib import Path

SFD_DIR = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def main():
    fa = SFD_DIR / "analysis" / "data" / "feature_analysis"
    candidates = load_jsonl(fa / "candidates.jsonl")
    evidence_d = {r["feature_idx"]: r for r in load_jsonl(fa / "candidate_evidence_discovery.jsonl")}
    evidence_c = {r["feature_idx"]: r for r in load_jsonl(fa / "candidate_evidence_confirmation.jsonl")}
    confound = {r["feature_idx"]: r for r in load_jsonl(fa / "confound_diagnostics.jsonl")}
    counterex = {r["feature_idx"]: r for r in load_jsonl(fa / "counterexamples.jsonl")}

    min_tokens, min_prompts = 5, 3
    out_path = SFD_DIR / "analysis" / "data" / "validation_results" / "review_bundles.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for cand in candidates:
            feat = cand["feature_idx"]
            conf_ev = evidence_c.get(feat, {})
            meets_min_evidence = (conf_ev.get("n_firing_tokens", 0) >= min_tokens
                                   and conf_ev.get("n_unique_firing_prompts", 0) >= min_prompts)
            bundle = {
                "feature_idx": feat, "dictionary_width": 8192, "seed": 0,
                "selection": cand,
                "discovery_evidence": evidence_d.get(feat, {}),
                "confirmation_evidence": conf_ev,
                "confirmation_meets_minimum_evidence_rule": meets_min_evidence,
                "confound_diagnostics": confound.get(feat, {}),
                "counterexamples": counterex.get(feat, {}).get("counterexamples", []),
            }
            f.write(json.dumps(bundle, ensure_ascii=False) + "\n")

    print(f"[bundles] wrote {out_path} ({len(candidates)} candidates)")


if __name__ == "__main__":
    main()
