"""Self-supervised pretraining of GRU encoder on raw tracklet trajectories."""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from pretrain_data import PretrainDataset
from trajectory_model import CrossingModel

DEVICE = "cuda"
EPOCHS = 30
BATCH_SIZE = 512
BASE_LR = 1e-3
MIN_LR = 1e-5
WARMUP_EPOCHS = 3


def compute_pretrain_loss(pred_traj, true_traj, frame_wh):
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
    return weighted.mean()


def get_lr(epoch):
    if epoch <= WARMUP_EPOCHS:
        return MIN_LR + (BASE_LR - MIN_LR) * epoch / WARMUP_EPOCHS
    remaining = EPOCHS - WARMUP_EPOCHS
    t = epoch - WARMUP_EPOCHS
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * t / remaining))


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("Building SSL dataset...")
    full_ds = PretrainDataset(augment=True)

    n_val = int(len(full_ds) * 0.1)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    val_ds_noaug = PretrainDataset(augment=False)
    val_indices = val_ds.indices

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=4, drop_last=True)
    val_sampler = SubsetRandomSampler(val_indices)
    val_dl = DataLoader(val_ds_noaug, batch_size=BATCH_SIZE, sampler=val_sampler,
                        pin_memory=True, num_workers=4)

    print(f"Train: {n_train}, Val: {n_val}")

    with open("model_config.json") as f:
        cfg = json.load(f)

    model = CrossingModel(**cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MIN_LR, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        loss_sum = 0.0
        n_batches = 0

        for seq, tgt_traj, frame_wh, cur_center in train_dl:
            seq = seq.to(DEVICE)
            tgt_traj = tgt_traj.to(DEVICE)
            frame_wh = frame_wh.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred_traj, _ = model(seq)
                loss = compute_pretrain_loss(pred_traj, tgt_traj, frame_wh)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for seq, tgt_traj, frame_wh, cur_center in val_dl:
                seq = seq.to(DEVICE)
                tgt_traj = tgt_traj.to(DEVICE)
                frame_wh = frame_wh.to(DEVICE)
                pred_traj, _ = model(seq)
                loss = compute_pretrain_loss(pred_traj, tgt_traj, frame_wh)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pretrained_full.pt")
            encoder_state = {}
            for k, v in model.state_dict().items():
                if k.startswith(("input_proj.", "layer_norm.", "gru.")):
                    encoder_state[k] = v
            torch.save(encoder_state, "pretrained_encoder.pt")
            marker = " *saved*"
        else:
            marker = ""

        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | LR: {lr:.6f}{marker}")

    elapsed = time.time() - t_start
    print(f"\nPretraining complete ({elapsed:.0f}s)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved: pretrained_encoder.pt, pretrained_full.pt")


if __name__ == "__main__":
    main()
