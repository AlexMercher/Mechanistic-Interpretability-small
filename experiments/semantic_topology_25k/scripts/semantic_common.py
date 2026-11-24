"""
Shared data-loading helpers for the semantic_topology_25k experiment.

Everything here is read-only against prior milestones:
  - normalized/*.jsonl                                  (Milestone 1)
  - experiments/duplicate_language_embedding/...         (Milestone 2)

No function in this module writes anywhere outside
experiments/semantic_topology_25k/.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
NORMALIZED_DIR = ROOT / "normalized"
PREV_EXPERIMENT_DIR = ROOT / "experiments" / "duplicate_language_embedding"
EMBEDDINGS_DIR = PREV_EXPERIMENT_DIR / "embeddings"
ANNOTATIONS_PATH = PREV_EXPERIMENT_DIR / "annotations" / "language_script_annotations.jsonl"
DUP_GROUPS_CSV = PREV_EXPERIMENT_DIR / "reports" / "exact_duplicate_groups.csv"
EXP_DIR = ROOT / "experiments" / "semantic_topology_25k"

DOMAINS = ["DNA", "EMOTIONS", "HINDI", "SCIENCE", "GANDHI"]


def load_normalized_records():
    """Returns dict[prompt_id] -> record, loaded from normalized/*.jsonl."""
    records = {}
    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["prompt_id"]] = rec
    return records


def load_annotations():
    """Returns dict[prompt_id] -> language/script annotation record."""
    annotations = {}
    with ANNOTATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            annotations[rec["prompt_id"]] = rec
    return annotations


def load_embeddings():
    """
    Returns (vectors, prompt_ids) where vectors[i] is the embedding for
    prompt_ids[i], in embedding_row order (0..N-1), taken directly from
    embedding_index.csv -- never re-derived or re-sorted independently of
    the manifest's own row assignment.
    """
    import numpy as np

    manifest = json.loads((EMBEDDINGS_DIR / "embedding_manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_DIR / "prompt_embeddings.npy")

    actual_sha256 = hashlib.sha256((EMBEDDINGS_DIR / "prompt_embeddings.npy").read_bytes()).hexdigest()
    if actual_sha256 != manifest["npy_sha256"]:
        raise RuntimeError(
            f"prompt_embeddings.npy sha256 mismatch vs manifest "
            f"(manifest={manifest['npy_sha256']} actual={actual_sha256}) -- refusing to use unverified embeddings"
        )

    with (EMBEDDINGS_DIR / "embedding_index.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prompt_ids = [None] * len(rows)
    for r in rows:
        prompt_ids[int(r["embedding_row"])] = r["prompt_id"]
    if any(p is None for p in prompt_ids):
        raise RuntimeError("embedding_index.csv has gaps in embedding_row -- not a clean permutation")

    return vectors, prompt_ids, manifest


def load_duplicate_group_map():
    """
    Returns dict[prompt_id] -> prompt_sha256 (recomputed hash group key) for
    every prompt_id that belongs to an exact-duplicate group (multiplicity > 1).
    Prompt_ids not in this dict are unique (their own singleton group).
    """
    group_of = {}
    with DUP_GROUPS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sha = row["prompt_sha256"]
            for pid in row["prompt_ids"].split(";"):
                group_of[pid] = sha
    return group_of


def unique_text_count(prompt_ids, group_of):
    """
    Count of distinct underlying prompt texts among prompt_ids, treating
    exact-duplicate-group membership as the same text and everything else
    as its own unique text.
    """
    seen_groups = set()
    seen_singletons = 0
    for pid in prompt_ids:
        g = group_of.get(pid)
        if g is None:
            seen_singletons += 1
        else:
            seen_groups.add(g)
    return len(seen_groups) + seen_singletons
