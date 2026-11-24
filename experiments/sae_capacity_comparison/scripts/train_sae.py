"""
Milestone 5, step 3: generic SAE trainer. Config+seed-driven; the pilot and
all real training runs (any of the three widths, any seed) go through this
one script so architecture/training code is guaranteed identical across the
capacity comparison.

Usage:
  python train_sae.py --config configs/sae_8192.json --seed 0 \
      --sparsity-coefficient 3e-4 --lr 3e-4 --batch-size 4096 --steps 20000

  python train_sae.py --config configs/sae_4096.json --seed 0 --pilot \
      --pilot-rows 50000 --steps 1500 --sparsity-coefficient 1e-4

Resumable: if checkpoints/<width>/seed_<k>/latest.pt exists, resumes from it
(same config required -- mismatches raise rather than silently continuing).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_common as sc  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent.parent


def build_lr_lambda(total_steps: int, warmup_frac: float = 0.05):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    return fn


@torch.no_grad()
def run_validation(model, val_x, preprocessor, device, batch_size, lam):
    model.eval()
    n = val_x.shape[0]
    total_recon = 0.0
    total_sparsity = 0.0
    total_ev_num = 0.0
    total_ev_den = 0.0
    l0_sum = 0.0
    for start in range(0, n, batch_size):
        raw = val_x[start:start + batch_size].to(device, torch.float32)
        x = preprocessor.transform(raw)
        x_hat, z = model(x)
        recon = ((x - x_hat) ** 2).sum(dim=1).sum().item()
        sparsity = z.abs().sum(dim=1).sum().item()
        total_recon += recon
        total_sparsity += sparsity
        total_ev_num += ((x - x_hat) ** 2).sum().item()
        total_ev_den += (x ** 2).sum().item()  # approx via sum of squares; refined in evaluate_reconstruction.py
        l0_sum += (z > 0).float().sum().item()
    model.train()
    return {
        "val_recon_mse_per_vector": total_recon / n,
        "val_sparsity_l1_per_vector": total_sparsity / n,
        "val_explained_variance_approx": 1.0 - total_ev_num / total_ev_den,
        "val_mean_l0": l0_sum / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--sparsity-coefficient", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-rows", type=int, default=50000)
    ap.add_argument("--amp", action="store_true",
                     help="torch.autocast('cuda', dtype=torch.float16) for the forward matmuls "
                          "+ GradScaler; loss/reduction stays fp32. Benchmark before relying on it.")
    ap.add_argument("--nan-check-every", type=int, default=20,
                     help="cheap single-scalar host sync to catch instability between full log_every reports")
    args = ap.parse_args()

    config = sc.load_config(args.config)
    width = config["dictionary_width"]
    input_dim = config["input_dim"]
    lr = args.lr or config["optimizer"]["lr"]
    batch_size = args.batch_size or config["batch_size"]
    lam = args.sparsity_coefficient

    sc.set_seed(args.seed)
    device = sc.get_device_no_fallback()
    print(f"[train_sae] device={device} width={width} seed={args.seed} pilot={args.pilot}")

    preprocessor = sc.Preprocessor.load(EXP_DIR / "preprocessing")

    max_rows = args.pilot_rows if args.pilot else None
    t0 = time.time()
    train_x = sc.load_split_activations("train", max_rows=max_rows)
    val_max_rows = max(2000, args.pilot_rows // 10) if args.pilot else None
    val_x = sc.load_split_activations("validation", max_rows=val_max_rows)
    print(f"[train_sae] loaded train={train_x.shape}, validation={val_x.shape} in {time.time()-t0:.1f}s")

    n_train = train_x.shape[0]

    model = sc.SparseAutoencoder(input_dim, width, seed=args.seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(args.steps))

    run_name = f"sae_{width}" + ("_pilot" if args.pilot else "")
    ckpt_dir = EXP_DIR / ("checkpoints_pilot" if args.pilot else "checkpoints") / f"sae_{width}" / f"seed_{args.seed}"
    log_dir = EXP_DIR / ("logs" if not args.pilot else EXP_DIR / "logs" / "pilot")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}_seed{args.seed}.jsonl"

    start_step = 0
    best_val_recon = float("inf")
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        start_step, extra = sc.load_checkpoint(latest_ckpt, model, optimizer)
        best_val_recon = extra.get("best_val_recon", float("inf"))
        print(f"[train_sae] resumed from step {start_step}")

    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_train, generator=gen)
    cursor = 0

    def next_batch_indices():
        nonlocal cursor, perm, gen
        if cursor + batch_size > n_train:
            perm = torch.randperm(n_train, generator=gen)
            cursor = 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        return idx

    torch.cuda.reset_peak_memory_stats()
    log_handle = log_path.open("a", encoding="utf-8")
    # GPU-resident -- no per-step .cpu() sync (a per-step host sync on a
    # dict_size-sized tensor was the dominant source of step-time overhead;
    # the dead-feature count is only materialized to Python at log_every).
    feature_last_active = torch.zeros(width, dtype=torch.long, device=device)
    dead_window = 2000
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    t_train_start = time.time()
    for step in range(start_step, args.steps):
        idx = next_batch_indices()
        raw = train_x[idx].to(device, torch.float32, non_blocking=True)
        x = preprocessor.transform(raw)

        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            x_hat, z = model(x)
        # loss/reduction always computed in fp32, per the task's explicit instruction,
        # regardless of whether the forward matmuls ran under autocast fp16.
        loss, recon, sparsity = sc.sae_loss(x, x_hat.float(), z.float(), lam)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.renormalize_decoder_()

        with torch.no_grad():
            active_mask = (z > 0).any(dim=0)  # GPU bool tensor, no host sync
            feature_last_active[active_mask] = step

        if step % args.nan_check_every == 0:
            # single-scalar host sync, cheap -- catches instability well before
            # the next full log_every report, without syncing every step.
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=1).mean().item()
                dead = int((step - feature_last_active > dead_window).sum().item())
                decoder_norm = model.W_d.norm(dim=0).mean().item()
                ev = sc.explained_variance(x.float(), x_hat.float())
                grad_sq = sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)
                grad_norm = float(grad_sq.sqrt().item())
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            record = {
                "step": step, "loss": loss.item(), "recon_mse_sum": recon.item(),
                "sparsity_l1_sum": sparsity.item(), "explained_variance": ev,
                "mean_l0": l0, "dead_features": dead, "grad_norm": grad_norm,
                "decoder_mean_norm": decoder_norm, "lr": scheduler.get_last_lr()[0],
                "peak_vram_gb": peak_vram,
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(f"[train_sae] step={step} loss={loss.item():.4f} recon={recon.item():.4f} "
                  f"l0={l0:.2f} dead={dead} ev={ev:.4f} vram={peak_vram:.2f}GB")

        if step % args.val_every == 0 or step == args.steps - 1:
            val_metrics = run_validation(model, val_x, preprocessor, device, batch_size, lam)
            val_metrics["step"] = step
            log_handle.write(json.dumps({"val": val_metrics}) + "\n")
            log_handle.flush()
            print(f"[train_sae] VAL step={step} {val_metrics}")
            if val_metrics["val_recon_mse_per_vector"] < best_val_recon:
                best_val_recon = val_metrics["val_recon_mse_per_vector"]
                sc.save_checkpoint(ckpt_dir / "best_val.pt", model, optimizer, step,
                                    {"best_val_recon": best_val_recon, "config": config, "args": vars(args)})

        if step % args.save_every == 0 or step == args.steps - 1:
            sc.save_checkpoint(latest_ckpt, model, optimizer, step,
                                {"best_val_recon": best_val_recon, "config": config, "args": vars(args)})

    sc.save_checkpoint(ckpt_dir / "final.pt", model, optimizer, args.steps,
                        {"best_val_recon": best_val_recon, "config": config, "args": vars(args)})
    log_handle.close()

    total_time = time.time() - t_train_start
    print(f"[train_sae] done. steps={args.steps} total_time_sec={total_time:.1f} "
          f"peak_vram_gb={torch.cuda.max_memory_allocated()/1e9:.2f}")


if __name__ == "__main__":
    main()
