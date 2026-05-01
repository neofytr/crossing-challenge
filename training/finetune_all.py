"""Finetune GRU models on ALL labeled data (train+eval+dev = 93,749 samples).

Used for blind final submission after dev-validated improvements are locked in.
No dev split → uses fixed epoch count instead of early stopping.
Requires pretrained_encoder.pt (from pretrain.py) to exist.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import sys
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from trajectory_data import TrajectoryDataset
from trajectory_model import CrossingModel

DEVICE = "cuda"
FIXED_EPOCHS = 25          # based on seed convergence: 10, 14, 20 → 25 is safe
BATCH_SIZE = 512
WARMUP_EPOCHS = 3
BASE_LR = 8e-4
MIN_LR = 1e-5
INTENT_WEIGHT = 50.0
MODEL_SEEDS = [42, 123, 456]
MODEL_CFG = {"input_dim": 14, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2}
DATA = _ROOT / "data"
PRETRAINED_ENCODER = Path(__file__).parent / "pretrained_encoder.pt"


def compute_loss(pred_traj, true_traj, pred_intent, true_intent, frame_wh):
    scale = frame_wh.unsqueeze(1)
    pred_px = pred_traj * scale
    true_px = true_traj * scale
    diff = pred_px - true_px
    huber_delta = 15.0
    abs_diff = torch.abs(diff)
    huber = torch.where(abs_diff < huber_delta,
                        0.5 * diff**2 / huber_delta,
                        abs_diff - 0.5 * huber_delta)
    huber_per_horizon = huber.sum(dim=-1)
    horizon_weights = torch.tensor([1.0, 1.0, 1.5, 2.0], device=huber_per_horizon.device)
    weighted = (huber_per_horizon * horizon_weights).sum(dim=-1) / horizon_weights.sum()
    traj_loss = weighted.mean()
    with torch.amp.autocast("cuda", enabled=False):
        intent_loss = F.binary_cross_entropy(pred_intent.float(), true_intent.float())
    return traj_loss + intent_loss * INTENT_WEIGHT


def get_lr(epoch):
    if epoch <= WARMUP_EPOCHS:
        return MIN_LR + (BASE_LR - MIN_LR) * epoch / WARMUP_EPOCHS
    remaining = FIXED_EPOCHS - WARMUP_EPOCHS
    t = epoch - WARMUP_EPOCHS
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * t / remaining))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_seed(seed: int, train_dl: DataLoader) -> None:
    output = str(_ROOT / f"best_model_s{seed}.pt")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = CrossingModel(**MODEL_CFG).to(DEVICE)

    if PRETRAINED_ENCODER.exists():
        pretrained = torch.load(PRETRAINED_ENCODER, map_location=DEVICE, weights_only=True)
        state = model.state_dict()
        loaded = sum(1 for k, v in pretrained.items() if k in state and not state.update({k: v}))
        model.load_state_dict(state)
        print(f"  Loaded {loaded} pretrained encoder tensors")

    encoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith(("traj_head.", "intent_head."))]
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith(("traj_head.", "intent_head."))]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": MIN_LR},
        {"params": head_params, "lr": MIN_LR},
    ], weight_decay=1e-4)
    scaler = GradScaler("cuda")

    t_start = time.time()
    for epoch in range(1, FIXED_EPOCHS + 1):
        lr = get_lr(epoch)
        optimizer.param_groups[0]["lr"] = lr * 0.25  # encoder at 0.25x
        optimizer.param_groups[1]["lr"] = lr

        model.train()
        loss_sum = 0.0
        n_batches = 0

        for seq, tgt_traj, tgt_intent, frame_wh, cur_center, cur_size in train_dl:
            seq = seq.to(DEVICE)
            tgt_traj = tgt_traj.to(DEVICE)
            tgt_intent = tgt_intent.squeeze(-1).to(DEVICE)
            frame_wh = frame_wh.to(DEVICE)

            if torch.rand(1).item() < 0.3:
                lam = np.random.beta(0.4, 0.4)
                idx = torch.randperm(seq.size(0), device=seq.device)
                seq = lam * seq + (1 - lam) * seq[idx]
                tgt_traj = lam * tgt_traj + (1 - lam) * tgt_traj[idx]
                tgt_intent = lam * tgt_intent + (1 - lam) * tgt_intent[idx]

            optimizer.zero_grad()
            with autocast("cuda"):
                pred_traj, pred_intent = model(seq)
                loss = compute_loss(pred_traj, tgt_traj, pred_intent, tgt_intent, frame_wh)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        print(f"  Epoch {epoch:3d}/{FIXED_EPOCHS} | Loss: {avg_loss:.3f} | LR: {lr:.6f}")

    torch.save(model.state_dict(), output)
    print(f"  Saved {output}  ({time.time()-t_start:.0f}s)")


def main():
    if not PRETRAINED_ENCODER.exists():
        raise SystemExit(
            f"pretrained_encoder.pt not found. Run pretrain.py first."
        )

    print(f"Loading train_all.parquet ({DATA / 'train_all.parquet'})...")
    train_ds = TrajectoryDataset(DATA / "train_all.parquet", augment=True)
    g = torch.Generator()
    g.manual_seed(0)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=4, drop_last=True,
                          worker_init_fn=seed_worker, generator=g)
    print(f"  {len(train_ds)} samples, {len(train_dl)} batches/epoch")

    for seed in MODEL_SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_one_seed(seed, train_dl)

    print("\nAll seeds complete. Run retrain_all.py next.")


if __name__ == "__main__":
    main()
