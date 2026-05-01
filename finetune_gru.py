from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from trajectory_data import build_dataloaders
from trajectory_model import CrossingModel

BCE_FLOOR = 0.2488
ADE_FLOOR = 49.80
DEVICE = "cuda"
EPOCHS = 40
PATIENCE = 12
BATCH_SIZE = 512
BASE_LR = 2e-4
MIN_LR = 1e-5
INTENT_WEIGHT = 50.0
CROSSING_WEIGHT = 3.0

MODEL_CFG = {"input_dim": 14, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2}


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

    sample_weight = 1.0 + true_intent.detach() * (CROSSING_WEIGHT - 1.0)
    traj_loss = (weighted * sample_weight).sum() / sample_weight.sum()

    with torch.amp.autocast("cuda", enabled=False):
        intent_loss = F.binary_cross_entropy(pred_intent.float(), true_intent.float())

    total_loss = traj_loss + intent_loss * INTENT_WEIGHT
    return total_loss, traj_loss, intent_loss


@torch.no_grad()
def evaluate(model, dev_dl):
    model.eval()
    all_pred_cx, all_pred_cy = [], []
    all_true_cx, all_true_cy = [], []
    all_pred_intent, all_true_intent = [], []

    for seq, tgt_traj, tgt_intent, frame_wh, cur_center, cur_size in dev_dl:
        seq = seq.to(DEVICE)
        cur_center = cur_center.to(DEVICE)
        pred_traj, pred_intent = model(seq)
        fwh_cpu = frame_wh
        cc_cpu = cur_center.cpu()
        pred_center_px = cc_cpu.unsqueeze(1) + pred_traj.cpu() * fwh_cpu.unsqueeze(1)
        true_center_px = cc_cpu.unsqueeze(1) + tgt_traj * fwh_cpu.unsqueeze(1)
        all_pred_cx.append(pred_center_px[:, :, 0])
        all_pred_cy.append(pred_center_px[:, :, 1])
        all_true_cx.append(true_center_px[:, :, 0])
        all_true_cy.append(true_center_px[:, :, 1])
        all_pred_intent.append(pred_intent.cpu())
        all_true_intent.append(tgt_intent.squeeze(-1))

    pred_cx = torch.cat(all_pred_cx, dim=0).numpy()
    pred_cy = torch.cat(all_pred_cy, dim=0).numpy()
    true_cx = torch.cat(all_true_cx, dim=0).numpy()
    true_cy = torch.cat(all_true_cy, dim=0).numpy()
    pred_int = torch.cat(all_pred_intent, dim=0).numpy()
    true_int = torch.cat(all_true_intent, dim=0).numpy()

    per_h_ade = []
    for h in range(4):
        ade_h = np.sqrt((pred_cx[:, h] - true_cx[:, h])**2 +
                        (pred_cy[:, h] - true_cy[:, h])**2).mean()
        per_h_ade.append(float(ade_h))
    mean_ade = float(np.mean(per_h_ade))

    pi = np.clip(pred_int, 1e-6, 1 - 1e-6)
    bce = -float(np.mean(true_int * np.log(pi) + (1 - true_int) * np.log(1 - pi)))

    composite = 0.5 * (bce / BCE_FLOOR) + 0.5 * (mean_ade / ADE_FLOOR)
    return composite, mean_ade, per_h_ade, bce


def get_lr(epoch):
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * epoch / EPOCHS))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--output", type=str, default="best_model_ft.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Fine-tuning {args.pretrained} → {args.output}")
    print(f"Crossing weight: {CROSSING_WEIGHT}x, LR: {BASE_LR}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dl, dev_dl = build_dataloaders(BATCH_SIZE, worker_init_fn=seed_worker, generator=g)

    model = CrossingModel(**MODEL_CFG).to(DEVICE)
    model.load_state_dict(torch.load(args.pretrained, map_location=DEVICE, weights_only=True))

    _, init_ade, init_per_h, init_bce = evaluate(model, dev_dl)
    h_str = " ".join(f"H{i+1}:{a:.1f}" for i, a in enumerate(init_per_h))
    print(f"Initial ADE: {init_ade:.1f} [{h_str}]  BCE: {init_bce:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    best_ade = init_ade
    best_epoch = 0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        total_loss_sum = 0.0
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
                loss, tl, il = compute_loss(pred_traj, tgt_traj, pred_intent, tgt_intent, frame_wh)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss_sum += loss.item()
            n_batches += 1

        avg_loss = total_loss_sum / max(n_batches, 1)
        composite, mean_ade, per_h_ade, bce = evaluate(model, dev_dl)

        h_str = " ".join(f"H{i+1}:{a:.1f}" for i, a in enumerate(per_h_ade))
        marker = " *" if mean_ade < best_ade else ""
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.2f} | "
              f"ADE: {mean_ade:.1f} [{h_str}] | BCE: {bce:.4f} | LR: {lr:.6f}{marker}")

        if mean_ade < best_ade:
            best_ade = mean_ade
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), args.output)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    elapsed = time.time() - t_start
    print(f"\nFINE-TUNING COMPLETE ({elapsed:.0f}s)")
    print(f"Best epoch: {best_epoch}, ADE: {best_ade:.1f} (was {init_ade:.1f})")
    print(f"Improvement: {init_ade - best_ade:.1f} px")


if __name__ == "__main__":
    main()
