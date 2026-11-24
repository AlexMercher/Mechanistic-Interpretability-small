"""
Milestone 5, step 2: freeze the preprocessing artifact that every SAE width
and seed will share.

Reads preprocessing/activation_statistics.json + mu_vector.pt (written by
inspect_activations.py from the train split only) and writes the frozen
preprocessing_config.json that sae_common.Preprocessor.load() reads at
train/eval time. Also sanity-checks the existing Milestone-4 train/validation
prompt-level split is usable as-is (no new split created), per the task's
instruction to reuse Milestone 4's split unless demonstrably invalid.
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
ROOT = EXP_DIR.parent.parent
EXTRACTION_DIR = ROOT / "experiments" / "activation_extraction_qwen3_1_7b"

STATS_PATH = EXP_DIR / "preprocessing" / "activation_statistics.json"
OUT_PATH = EXP_DIR / "preprocessing" / "preprocessing_config.json"


def main():
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    transform = stats["preprocessing_transform"]

    split_path = EXTRACTION_DIR / "metadata" / "prompt_split.jsonl"
    n_train_prompts = 0
    n_val_prompts = 0
    with split_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "train":
                n_train_prompts += 1
            elif row["split"] == "validation":
                n_val_prompts += 1

    config = {
        "method": transform["method"],
        "mu_vector_file": "mu_vector.pt",
        "scale": transform["scale"],
        "computed_from": "train split only (experiments/activation_extraction_qwen3_1_7b/activations/train)",
        "n_train_tokens_used": stats["n_train_tokens"],
        "input_dim": stats["input_dim"],
        "rationale": transform["rationale"],
        "reused_split_source": "experiments/activation_extraction_qwen3_1_7b (Milestone 4), seed 1337, val_fraction 0.1, prompt-level",
        "n_train_prompts": n_train_prompts,
        "n_validation_prompts": n_val_prompts,
        "no_new_split_created": True,
    }
    OUT_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[prepare_sae_data] wrote {OUT_PATH}")
    print(f"[prepare_sae_data] train prompts={n_train_prompts}, validation prompts={n_val_prompts}")
    print(f"[prepare_sae_data] scale={config['scale']:.6f}")


if __name__ == "__main__":
    main()
