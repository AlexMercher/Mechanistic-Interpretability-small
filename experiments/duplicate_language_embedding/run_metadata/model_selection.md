# Embedding Model Selection

**Date:** 2026-11-24
**Decision:** use `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, before downloading any weights.

## Hardware / environment check (done first, per task instructions)

- GPU: NVIDIA RTX 3050, 6 GB VRAM (`nvidia-smi` confirmed, driver 581.86, CUDA 13.0).
- System RAM: 16 GB.
- Free disk: 213 GB on `D:` (project drive) at time of writing -- comfortably more than the ~100 GB budget mentioned.
- `.venv` was empty (stdlib only) before this task; no embedding model or ML framework previously installed.
- HF/sentence-transformers caches are redirected into `weights/hf_cache/` for this project (env vars `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME` all set before any `transformers`/`sentence_transformers`/`huggingface_hub` import) -- nothing is cached under the default `C:\Users\...\.cache`.

## Candidates considered

| Model | License | Dim | Hindi coverage | Size (approx) | Notes |
|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` (chosen) | MIT | 384 | Explicit -- `hi` listed in the model card's supported-language set (102 languages, mC4-derived training data) | ~118M params, ~470 MB fp32 safetensors | Strong MTEB semantic-similarity/retrieval benchmarks for its size; requires a `"query: "` input prefix per the model card's usage convention. |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | 384 | Model card lists "50 languages" but does **not** explicitly enumerate Hindi among them | ~470 MB | Well-known, no input prefix needed, but Hindi coverage is not explicitly documented -- would need empirical verification before trusting it for this corpus. |
| `BAAI/bge-m3` | MIT | 1024 | Explicit broad multilingual coverage incl. Hindi | ~2.2 GB | Larger footprint and higher dimension for no clear benefit on short (15-57 word) prompts; rejected to avoid downloading multiple large models unnecessarily per task instructions. |

**Why `multilingual-e5-small` specifically:** it's the only one of the three with Hindi *explicitly* confirmed in its documented language coverage (not just "multilingual" in general), it's MIT-licensed (least restrictive), and at ~470 MB / 118M params it comfortably fits the 6 GB VRAM / 16 GB RAM budget with large headroom -- in fact small enough that CPU inference is also practical (see below). Only one model was downloaded.

## Hardware fit

The model is tiny by LLM standards (118M params, inference-only, no training). Estimated peak memory for a batch of our longest prompts (max 57 words) is well under 1 GB, whether on GPU or CPU. **Decision: run inference on CPU**, not GPU:
- Avoids installing a CUDA-enabled PyTorch wheel (2+ GB vs. ~200-300 MB for CPU-only), which is unnecessary weight for a model this size and a one-off 25,000-record batch job.
- 25,000 short-sentence embeddings on CPU with a 118M-param model is expected to take low single-digit minutes, not hours -- confirmed empirically in the pilot step before committing to the full run.
- The RTX 3050's 6 GB VRAM is more than sufficient if GPU were used instead; CPU was a complexity/footprint tradeoff, not a capability limitation.

## Known limitation (documented up front, not discovered after the fact)

Multilingual embedding models, including e5-small, are trained overwhelmingly on *native-script* text per language. Romanized Hindi / Hinglish (Latin-script, ~43% of the HINDI domain per `experiments/duplicate_language_embedding/reports/language_script_metrics.json`) is **not** a training regime these models are typically evaluated on. The model can still embed Romanized Hindi text (it won't error or truncate), but embedding *quality* for Romanized Hindi vs. Devanagari Hindi vs. English is an open question that the pilot step (`embedding_pilot_report.json`) checks empirically rather than assumes -- per task instructions, multilingual support does not guarantee equal-quality embeddings across scripts.

## Preprocessing / configuration decided here (documented, not changed silently later)

- Input text: the complete, original `prompt` field, verbatim -- no domain/subdomain/category labels appended, no truncation performed silently.
- Required `"query: "` prefix prepended per the e5 usage convention (applied uniformly to every prompt, not just a retrieval subset, since these are standalone semantic embeddings, not asymmetric query/passage pairs).
- Max sequence length: model supports up to 512 tokens; our longest prompt is 57 words (well under any token-length concern for this model, but the generation script still checks and records the condition explicitly per prompt rather than assuming).
- Normalization: embeddings L2-normalized (standard practice for this model family, makes cosine similarity equivalent to dot product for the downstream nearest-neighbour use case) -- recorded per-vector in `embedding_manifest.json`.
