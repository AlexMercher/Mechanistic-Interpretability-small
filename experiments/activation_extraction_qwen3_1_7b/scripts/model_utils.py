"""
Shared model-loading helpers for the activation_extraction_qwen3_1_7b
experiment. Importing this module redirects all Hugging Face caches into
weights/hf_cache/ at the project root -- model weights must never land in
the default C:\\Users\\...\\.cache (the project's C: drive is also, as of
this experiment, critically low on free space -- see run logs). This MUST
happen before `transformers`/`torch` is imported anywhere in the process,
so every script in this experiment does `import model_utils` first.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_CACHE_DIR = ROOT / "weights" / "hf_cache"
WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(WEIGHTS_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(WEIGHTS_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(WEIGHTS_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # keep resumable urllib3 downloads, not hf_transfer

MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_REVISION = None  # resolved and pinned to a commit sha at first load, see model_manifest.json


def get_device_and_check_vram(min_free_gb: float = 3.0):
    """Raises RuntimeError (never silently falls back to CPU) if CUDA is unavailable
    or if there is not enough free VRAM to plausibly fit the model."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available on this machine. This experiment requires GPU inference "
            "per the task's explicit instruction -- refusing to silently fall back to CPU."
        )
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB free VRAM on {props.name} (total {total_gb:.2f} GB) -- "
            f"below the {min_free_gb} GB safety floor for loading the model. Refusing to proceed "
            f"or fall back to CPU; free up GPU memory (close other GPU applications) and retry."
        )
    return "cuda", props.name, total_gb, free_gb


def load_model_and_tokenizer(dtype_name: str = "bfloat16", device: str = "cuda"):
    """Loads Qwen3-1.7B directly onto the GPU in the requested dtype, inference mode,
    gradient computation disabled. Returns (model, tokenizer, resolved_revision)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(WEIGHTS_CACHE_DIR))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(WEIGHTS_CACHE_DIR),
        torch_dtype=dtype,
        attn_implementation="sdpa",  # memory-efficient fused attention, broadly supported
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    resolved_revision = getattr(model.config, "_commit_hash", None)
    if not resolved_revision:
        # fall back to reading it off the local HF cache's snapshot directory name,
        # which is always the resolved commit sha regardless of transformers version behavior
        model_dir_name = "models--" + MODEL_NAME.replace("/", "--")
        snapshots_dir = WEIGHTS_CACHE_DIR / model_dir_name / "snapshots"
        if snapshots_dir.exists():
            candidates = [p.name for p in snapshots_dir.iterdir() if p.is_dir()]
            if len(candidates) == 1:
                resolved_revision = candidates[0]
            elif candidates:
                resolved_revision = sorted(candidates)[-1]  # ambiguous but deterministic; documented
    return model, tokenizer, resolved_revision
