"""
Milestone 6, Phase 2 step 3 (plan A4 / D.2.3): systematic counterexample generation,
Confirmation-subset ONLY (never Discovery -- counterexamples must test a hypothesis
formed on independent data, not the data that produced it).

Three search types per candidate, each restricted to Confirmation-subset prompts:
  1. same_concept_different_template: embedding-nearest-neighbors of the candidate's
     top Discovery-firing prompt, restricted to neighbors in a DIFFERENT template tier/
     cluster than the seed prompt. Tests whether the feature generalizes across
     phrasing for (apparently) the same content.
  2. different_domain_same_surface_form: embedding-nearest-neighbors restricted to a
     DIFFERENT domain than the seed (multilingual-e5-small embeddings are known, per
     Milestone 3, to be template-phrasing-driven more than meaning-driven -- so
     cross-domain high-similarity neighbors are often "same surface wording, different
     topic" cases, exactly what this check needs).
  3. same_script_unrelated_semantics: for candidates with a non-trivial Hindi-script
     firing share, Confirmation prompts with the same language_form but from a
     subdomain unrelated to any of the candidate's own firing subdomains.

For every counterexample prompt, the candidate feature's actual activation (max over
that prompt's Confirmation tokens) is measured directly -- not assumed -- so a
"counterexample" can either contradict or (less often) reinforce the working hypothesis;
both outcomes are recorded.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
WIDTH = 8192
SEED = 0
N_PER_TYPE = 5


def load_confirmation_prompt_ids():
    assignment = {}
    with (SFD_DIR / "configs" / "discovery_confirmation_assignment.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assignment[row["prompt_id"]] = row["subset"]
    return {pid for pid, s in assignment.items() if s == "confirmation"}


def build_embedding_index(confirmation_ids: set):
    """Returns the FULL embedding set (seed queries can be Discovery or Confirmation
    prompts) plus a boolean mask marking which rows are Confirmation-eligible as
    neighbor results."""
    prompt_ids, embeddings = sf.load_prompt_embeddings()
    conf_mask = np.array([pid in confirmation_ids for pid in prompt_ids])
    return prompt_ids, embeddings, conf_mask


def nearest_neighbors(seed_prompt_id, prompt_ids, embeddings, prompt_ids_to_embidx, conf_mask, k=50):
    """Seed can be any prompt (typically Discovery); returned neighbors are
    restricted to Confirmation-subset prompts only (conf_mask)."""
    if seed_prompt_id not in prompt_ids_to_embidx:
        return []
    seed_i = prompt_ids_to_embidx[seed_prompt_id]
    seed_vec = embeddings[seed_i:seed_i + 1]
    sims = (embeddings @ seed_vec.T).ravel()
    sims = np.where(conf_mask, sims, -np.inf)  # exclude non-Confirmation rows entirely
    order = np.argsort(-sims)[:k]
    return [(prompt_ids[i], float(sims[i])) for i in order if np.isfinite(sims[i]) and prompt_ids[i] != seed_prompt_id]


def main():
    candidates = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8")]
    evidence_discovery = {}
    with (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidate_evidence_discovery.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            evidence_discovery[row["feature_idx"]] = row

    records = sf.load_normalized_records()
    lang_annot = sf.load_language_script_annotations()
    template_tiers = {}
    with (SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "template_clusters.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            template_tiers[row["prompt_id"]] = (row["tier"], row["template_cluster_id"])

    confirmation_ids = load_confirmation_prompt_ids()
    all_prompt_ids, all_embeddings, conf_mask = build_embedding_index(confirmation_ids)
    prompt_ids_to_embidx = {pid: i for i, pid in enumerate(all_prompt_ids) if pid is not None}

    # Candidate -> list of (counterexample_type, prompt_id)
    test_requests = {}
    for cand in candidates:
        feat = cand["feature_idx"]
        ev = evidence_discovery.get(feat, {})
        top_examples = ev.get("examples", {}).get("top", [])
        if not top_examples:
            test_requests[feat] = []
            continue
        seed = top_examples[0]
        seed_pid = seed["prompt_id"]
        seed_tier, seed_cluster = template_tiers.get(seed_pid, (3, None))
        # Tier 3 means "template relationship UNKNOWN", not "same template as every
        # other Tier-3 prompt" -- (3, None) must not compare equal to another (3, None).
        # Use the prompt_id itself as the effective key for Tier 3 (same convention as
        # extract_candidate_evidence.py's one-per-template logic), so two different
        # Tier-3 prompts always count as different templates for this comparison.
        seed_key = (seed_tier, seed_cluster) if seed_tier in (1, 2) else (3, seed_pid)
        seed_domain = seed["domain"]

        neighbors = nearest_neighbors(seed_pid, all_prompt_ids, all_embeddings, prompt_ids_to_embidx, conf_mask, k=50)

        same_concept_diff_template = []
        diff_domain_same_surface = []
        for pid, sim in neighbors:
            rec = records[pid]
            tier, cluster = template_tiers.get(pid, (3, None))
            key = (tier, cluster) if tier in (1, 2) else (3, pid)
            if rec["domain"] == seed_domain and key != seed_key and len(same_concept_diff_template) < N_PER_TYPE:
                same_concept_diff_template.append((pid, sim))
            if rec["domain"] != seed_domain and len(diff_domain_same_surface) < N_PER_TYPE:
                diff_domain_same_surface.append((pid, sim))

        # same-script, unrelated-semantics: only meaningful if the candidate has a
        # non-trivial Hindi-script firing share in Discovery.
        same_script_unrelated = []
        dom_counts = {}
        for ex in ev.get("examples", {}).get("one_per_template", []):
            dom_counts[ex["subdomain"]] = dom_counts.get(ex["subdomain"], 0) + 1
        own_subdomains = set(dom_counts.keys())
        candidate_script_relevant = any(
            lang_annot.get(seed_pid, {}).get("language_form") in ("native_hindi", "romanized_hindi")
            for seed_pid in [seed_pid])
        if candidate_script_relevant:
            pool = [pid for pid in confirmation_ids
                    if lang_annot.get(pid, {}).get("language_form") in ("native_hindi", "romanized_hindi")
                    and records[pid]["subdomain"] not in own_subdomains]
            pool = sorted(pool)[:N_PER_TYPE]
            same_script_unrelated = [(pid, None) for pid in pool]

        reqs = [("same_concept_different_template", pid) for pid, _ in same_concept_diff_template]
        reqs += [("different_domain_same_surface_form", pid) for pid, _ in diff_domain_same_surface]
        reqs += [("same_script_unrelated_semantics", pid) for pid, _ in same_script_unrelated]
        test_requests[feat] = reqs

    # Gather union of (feature, prompt_id) pairs that need an activation measurement.
    needed_prompt_ids = set()
    for reqs in test_requests.values():
        needed_prompt_ids.update(pid for _, pid in reqs)
    print(f"[counterexamples] {len(needed_prompt_ids)} unique Confirmation prompts need measurement")

    meta_rows = []
    with sf.PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "validation" and row["prompt_id"] in needed_prompt_ids:
                meta_rows.append((row["activation_row_index"], row["prompt_id"]))
    meta_df = pd.DataFrame(meta_rows, columns=["activation_row_index", "prompt_id"]).sort_values("activation_row_index")

    device = sf.get_device_no_fallback()
    model, _, _ = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    full_val = sf.load_split_activations("validation")
    x_all = full_val[meta_df["activation_row_index"].to_numpy().copy()]
    del full_val

    feature_idxs = sorted({c["feature_idx"] for c in candidates})
    feat_idx_t = torch.tensor(feature_idxs, dtype=torch.long, device=device)
    feat_pos = {f: i for i, f in enumerate(feature_idxs)}

    max_activation = {}  # (feature_idx, prompt_id) -> max activation
    model.eval()
    with torch.no_grad():
        batch_size = 2048
        n = len(meta_df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            raw = x_all[start:end].to(device, torch.float32)
            x = preprocessor.transform(raw)
            z_sel = model.encode(x)[:, feat_idx_t].cpu().numpy()
            pids = meta_df["prompt_id"].to_numpy()[start:end]
            for i, pid in enumerate(pids):
                for f in feature_idxs:
                    val = float(z_sel[i, feat_pos[f]])
                    key = (f, pid)
                    if key not in max_activation or val > max_activation[key]:
                        max_activation[key] = val

    out_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / "counterexamples.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for cand in candidates:
            feat = cand["feature_idx"]
            results = []
            for ctype, pid in test_requests.get(feat, []):
                act = max_activation.get((feat, pid), 0.0)
                results.append({
                    "counterexample_type": ctype, "prompt_id": pid, "domain": records[pid]["domain"],
                    "subdomain": records[pid]["subdomain"], "max_activation_in_prompt": act,
                    "fired": act > 0.0, "prompt_text": records[pid]["prompt"][:400],
                    "subset": "confirmation",
                })
            f.write(json.dumps({"feature_idx": feat, "counterexamples": results}, ensure_ascii=False) + "\n")

    print(f"[counterexamples] wrote {out_path}")


if __name__ == "__main__":
    main()
