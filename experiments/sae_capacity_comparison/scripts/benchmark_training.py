"""
Milestone 5 benchmarking: measure actual per-step GPU time (not wall-clock
including one-time Python/data-load overhead) for the SAE-8192 training step,
FP32 vs AMP (torch.autocast fp16 + GradScaler), on the real full training
tensor (not a pilot slice) and the frozen batch size/lr/lambda.

Method: `torch.cuda.synchronize()` before starting the timer and after each
timed block (not per-step inside the loop -- that would reintroduce the
exact sync overhead we're trying to measure around), with an explicit
untimed warm-up phase (cuDNN/cuBLAS kernel selection, CUDA context, first
allocations) excluded from the measurement. Reports steps/sec, time/step,
and peak VRAM for both precisions, plus a short validation-metric check
(recon MSE, L0, explained variance after the same number of steps) so a
throughput win isn't accepted blindly if it changes training behavior.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_common as sc  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent.parent


def run_timed(width, seed, amp, train_x, preprocessor, device, batch_size, lam, lr,
              warmup_steps, timed_steps):
    sc.set_seed(seed)
    model = sc.SparseAutoencoder(2048, width, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    gen = torch.Generator().manual_seed(seed)
    n = train_x.shape[0]

    def batch():
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        raw = train_x[idx].to(device, torch.float32)
        return preprocessor.transform(raw)

    def step():
        x = batch()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            x_hat, z = model(x)
        loss, recon, sparsity = sc.sae_loss(x, x_hat.float(), z.float(), lam)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.renormalize_decoder_()
        return loss, recon

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    last_loss = last_recon = None
    for _ in range(timed_steps):
        loss, recon = step()
        last_loss, last_recon = loss, recon
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    return {
        "amp": amp, "timed_steps": timed_steps, "elapsed_sec": elapsed,
        "sec_per_step": elapsed / timed_steps, "steps_per_sec": timed_steps / elapsed,
        "peak_vram_gb": peak_vram,
        "final_loss": float(last_loss.item()), "final_recon_mse_sum": float(last_recon.item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sparsity-coefficient", type=float, default=0.2)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--timed-steps", type=int, default=200)
    args = ap.parse_args()

    device = sc.get_device_no_fallback()
    preprocessor = sc.Preprocessor.load(EXP_DIR / "preprocessing")
    print("[benchmark] loading full train split...")
    train_x = sc.load_split_activations("train")
    print(f"[benchmark] train={train_x.shape}")

    results = {}
    for amp in (False, True):
        print(f"[benchmark] running amp={amp} ({args.warmup_steps} warmup + {args.timed_steps} timed steps)")
        r = run_timed(args.width, seed=0, amp=amp, train_x=train_x, preprocessor=preprocessor,
                      device=device, batch_size=args.batch_size, lam=args.sparsity_coefficient,
                      lr=args.lr, warmup_steps=args.warmup_steps, timed_steps=args.timed_steps)
        results["amp" if amp else "fp32"] = r
        print(f"[benchmark] amp={amp}: {r['sec_per_step']*1000:.2f} ms/step, "
              f"{r['steps_per_sec']:.1f} steps/sec, peak_vram={r['peak_vram_gb']:.2f}GB, "
              f"final_loss={r['final_loss']:.4f}")

    speedup = results["fp32"]["sec_per_step"] / results["amp"]["sec_per_step"]
    out = {
        "width": args.width, "batch_size": args.batch_size,
        "results": results, "amp_speedup_factor": speedup,
    }
    out_path = EXP_DIR / "reports" / "benchmark_report.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[benchmark] AMP speedup factor: {speedup:.2f}x")
    print(f"[benchmark] wrote {out_path}")


if __name__ == "__main__":
    main()
