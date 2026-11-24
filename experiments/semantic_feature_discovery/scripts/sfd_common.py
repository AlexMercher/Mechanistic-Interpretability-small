"""
Shared code for Milestone 6 (semantic_feature_discovery). Reuses Milestone 5's
SAE loading/preprocessing code read-only (never edited -- M5 is frozen) and adds
M6-specific loaders for corpus/language/duplicate/embedding artifacts that M5
never needed.

Import order matters: this module injects Milestone 5's scripts/ directory onto
sys.path so `sae_common` resolves, then re-exports the pieces M6 needs.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SFD_DIR = Path(__file__).resolve().parent.parent          # experiments/semantic_feature_discovery
ROOT = SFD_DIR.parent.parent                                # project root
M5_SCRIPTS_DIR = ROOT / "experiments" / "sae_capacity_comparison" / "scripts"
M4_DIR = ROOT / "experiments" / "activation_extraction_qwen3_1_7b"
M2_DIR = ROOT / "experiments" / "duplicate_language_embedding"
M2_VALIDATION_DIR = ROOT / "experiments" / "corpus_validation_25k"
M3_DIR = ROOT / "experiments" / "semantic_topology_25k"
NORMALIZED_DIR = ROOT / "normalized"

sys.path.insert(0, str(M5_SCRIPTS_DIR))
import sae_common as sc  # noqa: E402  -- M5's frozen module, read-only reuse

# Re-exported for convenience so M6 scripts only need `import sfd_common as sf`
SparseAutoencoder = sc.SparseAutoencoder
Preprocessor = sc.Preprocessor
load_split_activations = sc.load_split_activations
run_validation_pass = sc.run_validation_pass
load_model_for_eval = sc.load_model_for_eval
build_domain_ids = sc.build_domain_ids
load_domain_labels = sc.load_domain_labels
get_device_no_fallback = sc.get_device_no_fallback
set_seed = sc.set_seed
EXTRACTION_DIR = sc.EXTRACTION_DIR
PROMPT_TOKEN_METADATA_PATH = sc.PROMPT_TOKEN_METADATA_PATH
SHARD_MANIFEST_PATH = sc.SHARD_MANIFEST_PATH


def load_normalized_records() -> dict:
    """prompt_id -> full normalized record, across all 5 domains."""
    records = {}
    for path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["prompt_id"]] = rec
    return records


def load_language_script_annotations() -> dict:
    """prompt_id -> {domain, subdomain, category, language, script, language_form,
    classification_confidence, ...}."""
    path = M2_DIR / "annotations" / "language_script_annotations.jsonl"
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["prompt_id"]] = row
    return out


def load_near_duplicate_candidates() -> list:
    """List of dicts: domain, prompt_id_a, prompt_id_b, category, subdomain,
    similarity_ratio (float). Source: M2's near-duplicate audit (SequenceMatcher
    >=0.72, blocked by (category, subdomain))."""
    path = M2_VALIDATION_DIR / "reports" / "near_duplicate_candidates.csv"
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["similarity_ratio"] = float(row["similarity_ratio"])
            rows.append(row)
    return rows


def load_exact_duplicate_groups() -> list:
    """List of dicts: prompt_sha256, multiplicity (int), prompt_ids (list[str]),
    domains, subdomains, categories, prompt_text."""
    path = M2_DIR / "reports" / "exact_duplicate_groups.csv"
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["multiplicity"] = int(row["multiplicity"])
            row["prompt_ids"] = row["prompt_ids"].split(";")
            rows.append(row)
    return rows


def load_prompt_embeddings():
    """Returns (prompt_id_list, embeddings ndarray [N, 384]) aligned by row index,
    read from M2's embedding_index.csv + prompt_embeddings.npy. Read-only."""
    index_path = M2_DIR / "embeddings" / "embedding_index.csv"
    npy_path = M2_DIR / "embeddings" / "prompt_embeddings.npy"
    ids_by_row = {}
    with index_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                ids_by_row[int(row["embedding_row"])] = row["prompt_id"]
    embeddings = np.load(npy_path)
    prompt_ids = [ids_by_row.get(i) for i in range(embeddings.shape[0])]
    return prompt_ids, embeddings


def load_validation_prompt_ids() -> list:
    """All validation-split prompt_ids (2,533), from M4's prompt_split.jsonl --
    the frozen, unmodified Milestone-4 split. Never altered by M6."""
    path = EXTRACTION_DIR / "metadata" / "prompt_split.jsonl"
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "validation":
                ids.append(row["prompt_id"])
    return ids
