"""
Shared helpers for embedding generation/validation scripts. Importing this
module redirects all Hugging Face / sentence-transformers caches into
weights/hf_cache/ at the project root -- model weights must never land in
the default C:\\Users\\...\\.cache location. This MUST happen before
`sentence_transformers`/`transformers`/`torch` is imported anywhere in the
process, so other scripts should `import embed_utils` first.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_CACHE_DIR = ROOT / "weights" / "hf_cache"
WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(WEIGHTS_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(WEIGHTS_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(WEIGHTS_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(WEIGHTS_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "  # required by the e5 usage convention, see run_metadata/model_selection.md


def load_model(model_name: str = MODEL_NAME, revision: str = MODEL_REVISION, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    print(f"[embed_utils] loading {model_name}@{revision} on {device}, "
          f"cache={os.environ['HF_HOME']}")
    model = SentenceTransformer(model_name, revision=revision, device=device, cache_folder=str(WEIGHTS_CACHE_DIR))
    return model


def token_length(model, text: str) -> int:
    """Token count as the model's own tokenizer sees it (includes the query prefix)."""
    enc = model.tokenizer(text, truncation=False)
    return len(enc["input_ids"])


def embed_texts(model, texts, batch_size: int = 64, prefix: str = QUERY_PREFIX):
    """Returns (vectors: np.ndarray[N, dim] float32 L2-normalized, token_counts: list[int])."""
    import numpy as np
    prefixed = [prefix + t for t in texts]
    token_counts = [token_length(model, t) for t in prefixed]
    vectors = model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return vectors.astype(np.float32), token_counts
