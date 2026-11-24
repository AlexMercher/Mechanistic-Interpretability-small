"""
Shared code for every script in experiments/sae_capacity_comparison/:
activation loading (from the frozen Milestone-4 shards, read-only), the
SAE architecture, preprocessing transform, and checkpoint I/O.

Data-loading design note: the full train split (895,502 x 2048 vectors) fits
comfortably in system RAM at bf16 (~3.67 GB of the 16 GB available), so each
shard is read from disk ONCE at loader construction and concatenated into a
single CPU-resident bf16 tensor; GPU batches are then sliced from that
CPU tensor and upcast to fp32 only after the per-batch `.to(device)` copy.
This never puts the full dataset on GPU (only one batch at a time), and
avoids re-reading shards from disk on every one of the ~50-100 shuffled
epochs needed to get enough training signal out of ~895K vectors -- true
per-step disk streaming would thrash random-access reads against
safetensors files for a dataset that already fits in RAM. This is a
deliberate, documented adaptation of the task's "stream from disk" guidance
to this project's actual data volume and hardware (16GB system RAM, 6GB VRAM).
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
ROOT = EXP_DIR.parent.parent
EXTRACTION_DIR = ROOT / "experiments" / "activation_extraction_qwen3_1_7b"
ACTIVATIONS_DIR = EXTRACTION_DIR / "activations"
SHARD_MANIFEST_PATH = EXTRACTION_DIR / "metadata" / "shard_manifest.json"
PROMPT_TOKEN_METADATA_PATH = EXTRACTION_DIR / "metadata" / "prompt_token_metadata.jsonl"

INPUT_DIM = 2048


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device_no_fallback():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. SAE training in this milestone requires GPU "
            "per the task's explicit instruction -- refusing to silently fall back to CPU."
        )
    return "cuda"


def load_config(config_path: Path) -> dict:
    """Merges a per-width config (sae_4096.json etc.) over common_config.json."""
    config_path = Path(config_path)
    common_path = config_path.parent / "common_config.json"
    common = json.loads(common_path.read_text(encoding="utf-8"))
    specific = json.loads(config_path.read_text(encoding="utf-8"))
    merged = {**common, **specific}
    return merged


def verify_and_load_shard(path: Path, expected_sha256: str, expected_n_rows: int) -> torch.Tensor:
    import hashlib
    from safetensors import safe_open

    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha256:
        raise RuntimeError(f"shard {path} sha256 mismatch -- refusing to train on unverified data")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        x = f.get_tensor("activations")
    if x.shape[0] != expected_n_rows or x.shape[1] != INPUT_DIM:
        raise RuntimeError(f"shard {path} shape mismatch: {x.shape}")
    return x


def load_split_activations(split: str, max_rows: int = None) -> torch.Tensor:
    """Loads every shard of one split (train or validation) into one
    concatenated CPU bf16 tensor, verifying each shard's sha256 against the
    manifest first. `max_rows` truncates (used by the pilot for a fast slice)."""
    manifest = json.loads(SHARD_MANIFEST_PATH.read_text(encoding="utf-8"))
    shards = manifest[split]["shards"]
    pieces = []
    total = 0
    for s in shards:
        path = ACTIVATIONS_DIR / s["file"]
        x = verify_and_load_shard(path, s["sha256"], s["n_rows"])
        pieces.append(x)
        total += x.shape[0]
        if max_rows is not None and total >= max_rows:
            break
    full = torch.cat(pieces, dim=0)
    if max_rows is not None:
        full = full[:max_rows]
    return full.contiguous()


def load_domain_labels(split: str, n_rows: int) -> list:
    """Returns a list of (domain, subdomain, prompt_id) indexed by
    activation_row_index for the given split, read from
    prompt_token_metadata.jsonl. Used only for evaluation/interpretability,
    never for training."""
    labels = [None] * n_rows
    with PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != split:
                continue
            idx = row["activation_row_index"]
            if idx < n_rows:
                labels[idx] = (row["domain"], row["subdomain"], row["prompt_id"], row["token_position"])
    return labels


class Preprocessor:
    """Frozen (mu, scale) transform: y = scale * (x - mu). Computed once on
    train data by inspect_activations.py / prepare_sae_data.py, then applied
    identically everywhere (train and validation, all three widths)."""

    def __init__(self, mu: torch.Tensor, scale: float):
        self.mu = mu
        self.scale = float(scale)

    @classmethod
    def load(cls, preprocessing_dir: Path):
        cfg = json.loads((preprocessing_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
        mu = torch.load(preprocessing_dir / "mu_vector.pt")
        return cls(mu, cfg["scale"])

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu.to(x.device, x.dtype)) * self.scale

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        return y / self.scale + self.mu.to(y.device, y.dtype)


class SparseAutoencoder(torch.nn.Module):
    """z = ReLU(W_e @ (x - b_d) + b_e); x_hat = W_d @ z + b_d.

    Decoder columns (W_d's columns, i.e. dictionary directions) are kept at
    unit L2 norm by renormalizing after every optimizer step (caller's
    responsibility -- see renormalize_decoder_ below), which prevents the
    decoder from trivially shrinking its own norm to minimize the L1
    sparsity penalty on z without actually reducing active-feature count.
    Encoder is initialized as decoder^T (tied at step 0 only, then trained
    independently) -- a standard SAE init that starts the encoder/decoder
    in a self-consistent configuration rather than from pure noise.
    """

    def __init__(self, input_dim: int, dict_size: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        decoder_init = torch.randn(input_dim, dict_size, generator=gen)
        decoder_init = decoder_init / decoder_init.norm(dim=0, keepdim=True)

        self.W_d = torch.nn.Parameter(decoder_init.clone())
        self.W_e = torch.nn.Parameter(decoder_init.t().clone())
        self.b_e = torch.nn.Parameter(torch.zeros(dict_size))
        self.b_d = torch.nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_d) @ self.W_e.t() + self.b_e)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_d.t() + self.b_d

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def renormalize_decoder_(self):
        norms = self.W_d.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_d.div_(norms)


def sae_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, lam: float):
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()
    sparsity = z.abs().sum(dim=1).mean()
    total = recon + lam * sparsity
    return total, recon, sparsity


def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """1 - Var(x - x_hat) / Var(x), variance taken over all elements (both
    axes), matching the usual SAE-literature definition. Computed in the
    same (preprocessed) space as training -- the FINAL_REPORT must state
    this explicitly to distinguish raw vs. transformed-space metrics."""
    resid_var = (x - x_hat).var(unbiased=False)
    total_var = x.var(unbiased=False)
    return float(1.0 - resid_var / total_var)


def load_model_for_eval(width: int, seed: int, checkpoint_name: str = "best_val", device: str = "cuda"):
    """Loads a trained SAE for evaluation (seed's checkpoint from the real,
    non-pilot checkpoints/ tree). Returns (model, config)."""
    ckpt_path = EXP_DIR / "checkpoints" / f"sae_{width}" / f"seed_{seed}" / f"{checkpoint_name}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["extra"]["config"]
    model = SparseAutoencoder(config["input_dim"], width, seed=seed).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config, ckpt


@torch.no_grad()
def run_validation_pass(model, val_x: torch.Tensor, domain_ids, n_domains: int,
                         preprocessor: "Preprocessor", device: str, batch_size: int = 4096):
    """Single forward pass over the full validation split, accumulating every
    raw statistic the reconstruction/sparsity/utilization evaluation scripts
    need -- so all three read off the exact same held-out activation vectors
    and the same single pass (no risk of drift between eval scripts)."""
    n = val_x.shape[0]
    width = model.W_e.shape[0]

    per_row_mse = np.zeros(n, dtype=np.float64)
    per_row_l0 = np.zeros(n, dtype=np.int32)
    per_row_l1 = np.zeros(n, dtype=np.float64)
    feature_active_count = np.zeros(width, dtype=np.int64)
    feature_activation_sum = np.zeros(width, dtype=np.float64)
    domain_sq_error = np.zeros(n_domains, dtype=np.float64)
    domain_sq_x = np.zeros(n_domains, dtype=np.float64)
    domain_count = np.zeros(n_domains, dtype=np.int64)
    sum_sq_error = 0.0
    sum_sq_x = 0.0

    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        raw = val_x[start:end].to(device, torch.float32)
        x = preprocessor.transform(raw)
        x_hat, z = model(x)
        diff2 = (x - x_hat) ** 2
        row_sq_err = diff2.sum(dim=1)
        row_sq_x = (x ** 2).sum(dim=1)
        row_l0 = (z > 0).sum(dim=1)
        row_l1 = z.abs().sum(dim=1)

        per_row_mse[start:end] = row_sq_err.cpu().numpy()
        per_row_l0[start:end] = row_l0.cpu().numpy()
        per_row_l1[start:end] = row_l1.cpu().numpy()
        feature_active_count += (z > 0).sum(dim=0).cpu().numpy()
        feature_activation_sum += z.sum(dim=0).cpu().numpy()
        sum_sq_error += row_sq_err.sum().item()
        sum_sq_x += row_sq_x.sum().item()

        ids_batch = domain_ids[start:end]
        np.add.at(domain_sq_error, ids_batch, row_sq_err.cpu().numpy())
        np.add.at(domain_sq_x, ids_batch, row_sq_x.cpu().numpy())
        np.add.at(domain_count, ids_batch, 1)

    return {
        "n_rows": n, "width": width,
        "per_row_mse": per_row_mse, "per_row_l0": per_row_l0, "per_row_l1": per_row_l1,
        "feature_active_count": feature_active_count, "feature_activation_sum": feature_activation_sum,
        "domain_sq_error": domain_sq_error, "domain_sq_x": domain_sq_x, "domain_count": domain_count,
        "sum_sq_error": sum_sq_error, "sum_sq_x": sum_sq_x,
    }


def build_domain_ids(domain_labels: list):
    """domain_labels: list of (domain, subdomain, prompt_id, token_position) or None,
    indexed by activation_row_index (from sae_common.load_domain_labels).
    Returns (domain_ids: np.array[int], domain_list: list[str]). Any row with a
    missing label is mapped to an explicit trailing "UNKNOWN" bucket -- never -1,
    which would alias to the last real domain via numpy negative indexing."""
    domains = sorted({lab[0] for lab in domain_labels if lab is not None})
    has_unknown = any(lab is None for lab in domain_labels)
    if has_unknown:
        domains = domains + ["UNKNOWN"]
    domain_to_id = {d: i for i, d in enumerate(domains)}
    unknown_id = domain_to_id.get("UNKNOWN", 0)
    ids = np.array([domain_to_id[lab[0]] if lab is not None else unknown_id for lab in domain_labels])
    return ids, domains


def save_checkpoint(path: Path, model: SparseAutoencoder, optimizer, step: int, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "extra": extra,
    }, path)


def load_checkpoint(path: Path, model: SparseAutoencoder, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("extra", {})
