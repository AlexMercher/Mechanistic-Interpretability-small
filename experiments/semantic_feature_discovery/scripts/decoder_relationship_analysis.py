"""
Milestone 6, Phase 4 step 2 (plan B.2 decoder-space relationships): compares FOUR
genuinely different kinds of evidence between every pair of the 61 eligible candidates
-- decoder-direction similarity, activation-pattern similarity, top-example (prompt)
overlap, and reviewer-interpretation lexical overlap -- and explicitly flags cases
where they disagree. Decoder cosine similarity is used as a diagnostic signal only,
never treated as a semantic verdict on its own (per the task's explicit instruction).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
REL_DIR = SFD_DIR / "analysis" / "data" / "relationship_analysis"
WIDTH = 8192
SEED = 0
TOP_PROMPT_K = 20
DECODER_HIGH = 0.3
ACTIVATION_HIGH = 0.3
OVERLAP_HIGH = 0.1
WORD_RE = re.compile(r"[a-zA-Z]+")
STOPWORDS = {"the", "a", "an", "is", "of", "to", "and", "in", "on", "for", "this", "that",
             "with", "fires", "feature", "firing", "it", "its", "as", "at", "or", "be", "not"}


def interpretation_words(review):
    text = (review.get("candidate_interpretation", "") + " " + review.get("short_label", "")).lower()
    words = {w for w in WORD_RE.findall(text) if w not in STOPWORDS and len(w) > 2}
    return words


@torch.no_grad()
def main():
    eligible = json.loads((REL_DIR / "eligible_candidates.json").read_text())["candidates"]

    device = sf.get_device_no_fallback()
    model, _, _ = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    Wd = model.W_d.detach()[:, eligible]  # [2048, 61], unit-norm columns
    decoder_sim = (Wd.t() @ Wd).cpu().numpy()  # [61, 61]

    prompt_df = pd.read_pickle(REL_DIR / "prompt_activation_matrix_confirmation.pkl")
    feat_cols = [f"f{c}" for c in eligible]
    act_mat = prompt_df[feat_cols].to_numpy()
    act_norm = act_mat / (np.linalg.norm(act_mat, axis=0, keepdims=True) + 1e-8)
    activation_sim = act_norm.T @ act_norm  # [61, 61]

    top_prompts = {}
    for i, feat in enumerate(eligible):
        col = act_mat[:, i]
        order = np.argsort(-col)[:TOP_PROMPT_K]
        top_prompts[feat] = set(prompt_df.index.to_numpy()[order])

    reviews = {r["feature_idx"]: r for r in
               [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "primary_reviews.jsonl").open(encoding="utf-8")]}
    interp_words = {feat: interpretation_words(reviews[feat]) for feat in eligible}

    pairs = []
    disagreements = []
    for i, a in enumerate(eligible):
        for j, b in enumerate(eligible):
            if j <= i:
                continue
            dec = float(decoder_sim[i, j])
            act = float(activation_sim[i, j])
            jac_prompts = len(top_prompts[a] & top_prompts[b]) / max(len(top_prompts[a] | top_prompts[b]), 1)
            wa, wb = interp_words[a], interp_words[b]
            jac_interp = len(wa & wb) / max(len(wa | wb), 1) if (wa or wb) else 0.0

            record = {"feature_a": a, "feature_b": b, "decoder_cosine": dec,
                      "activation_pattern_cosine": act, "top_prompt_jaccard": jac_prompts,
                      "interpretation_lexical_jaccard": jac_interp}
            pairs.append(record)

            decoder_high = dec >= DECODER_HIGH
            others_high = (act >= ACTIVATION_HIGH) or (jac_prompts >= OVERLAP_HIGH) or (jac_interp >= OVERLAP_HIGH)
            if decoder_high and not others_high:
                disagreements.append({**record, "disagreement_type": "decoder_high_semantic_low",
                                       "note": "Decoder directions are similar but activation pattern, top-prompt overlap, and interpretation text all disagree -- decoder similarity alone would overstate a relationship here."})
            elif not decoder_high and (act >= ACTIVATION_HIGH and jac_prompts >= OVERLAP_HIGH):
                disagreements.append({**record, "disagreement_type": "semantic_high_decoder_low",
                                       "note": "Activation pattern and top-prompt overlap both agree but decoder directions are not similar -- two functionally-related features need not share a decoder direction."})

    out = {
        "eligible_candidates": eligible,
        "thresholds": {"decoder_high": DECODER_HIGH, "activation_high": ACTIVATION_HIGH, "overlap_high": OVERLAP_HIGH},
        "pairs": pairs,
        "disagreements": disagreements,
        "methodology_note": ("Four distinct evidence types, never collapsed into one score: "
                              "(1) decoder_cosine = geometric similarity of learned dictionary "
                              "directions (a diagnostic, not a semantic verdict); (2) "
                              "activation_pattern_cosine = functional co-activation across "
                              "Confirmation-subset prompts; (3) top_prompt_jaccard = overlap of "
                              "each feature's own top-20 firing prompts; (4) "
                              "interpretation_lexical_jaccard = word overlap between the two "
                              "features' independent primary-reviewer interpretations (a crude, "
                              "deterministic proxy for reviewer-judged semantic similarity, not a "
                              "claim of NLP-level understanding)."),
    }
    out_path = REL_DIR / "decoder_relationship_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[decoder_rel] {len(pairs)} pairs, {len(disagreements)} decoder-vs-semantic disagreements")
    print(f"[decoder_rel] wrote {out_path}")


if __name__ == "__main__":
    main()
