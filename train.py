from __future__ import annotations

import json
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
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 256


def compute_loss(pred_traj, true_traj, pred_intent, true_intent, frame_wh):
    scale = frame_wh.unsqueeze(1)
    pred_px = pred_traj * scale
    true_px = true_traj * scale

    diff = pred_px - true_px
    ade_per_horizon = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)

    horizon_weights = torch.tensor([1.0, 1.0, 1.5, 2.0], device=ade_per_horizon.device)
    weighted_ade = (ade_per_horizon * horizon_weights).sum(dim=-1) / horizon_weights.sum()
    traj_loss = weighted_ade.mean()

    with torch.amp.autocast("cuda", enabled=False):
        intent_loss = F.binary_cross_entropy(pred_intent.float(), true_intent.float())

    total_loss = traj_loss + intent_loss * 50.0
    return total_loss, traj_loss, intent_loss


@torch.no_grad()
def evaluate(model, dev_dl):
    model.eval()
    all_pred_cx = []
    all_pred_cy = []
    all_true_cx = []
    all_true_cy = []
    all_pred_intent = []
    all_true_intent = []

    for seq, tgt_traj, tgt_intent, frame_wh, cur_center, cur_size in dev_dl:
        seq = seq.to(DEVICE)
        frame_wh = frame_wh.to(DEVICE)
        cur_center = cur_center.to(DEVICE)

        pred_traj, pred_intent = model(seq)

        scale = frame_wh.unsqueeze(1)
        pred_center_px = cur_center.cpu().unsqueeze(1) + pred_traj.cpu() * frame_wh.cpu().unsqueeze(1)
        true_center_px = cur_center.cpu().unsqueeze(1) + tgt_traj * frame_wh.cpu().unsqueeze(1)

        all_pred_cx.append(pred_center_px[:, :, 0].cpu())
        all_pred_cy.append(pred_center_px[:, :, 1].cpu())
        all_true_cx.append(true_center_px[:, :, 0].cpu())
        all_true_cy.append(true_center_px[:, :, 1].cpu())
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
        ade_h = np.sqrt((pred_cx[:, h] - true_cx[:, h]) ** 2 +
                        (pred_cy[:, h] - true_cy[:, h]) ** 2).mean()
        per_h_ade.append(float(ade_h))
    mean_ade = float(np.mean(per_h_ade))

    pi = np.clip(pred_int, 1e-6, 1 - 1e-6)
    bce = -float(np.mean(true_int * np.log(pi) + (1 - true_int) * np.log(1 - pi)))

    composite = 0.5 * (bce / BCE_FLOOR) + 0.5 * (mean_ade / ADE_FLOOR)
    return composite, mean_ade, per_h_ade, bce


def main():
    print("Building dataloaders...")
    train_dl, dev_dl = build_dataloaders(BATCH_SIZE)
    print(f"Train batches: {len(train_dl)}, Dev batches: {len(dev_dl)}")

    model = CrossingModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda")

    best_composite = float("inf")
    best_epoch = -1
    best_ade = 0.0
    best_per_h = []
    best_bce = 0.0
    patience_counter = 0

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss_sum = 0.0
        n_batches = 0

        for seq, tgt_traj, tgt_intent, frame_wh, cur_center, cur_size in train_dl:
            seq = seq.to(DEVICE)
            tgt_traj = tgt_traj.to(DEVICE)
            tgt_intent = tgt_intent.squeeze(-1).to(DEVICE)
            frame_wh = frame_wh.to(DEVICE)

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

        scheduler.step()
        avg_loss = total_loss_sum / max(n_batches, 1)

        composite, mean_ade, per_h_ade, bce = evaluate(model, dev_dl)
        lr = optimizer.param_groups[0]["lr"]

        h_str = ", ".join(f"H{i+1}: {a:.1f}" for i, a in enumerate(per_h_ade))
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_loss:.2f} | "
              f"Dev Composite: {composite:.4f} | ADE: {mean_ade:.1f} px [{h_str}] | "
              f"BCE: {bce:.4f} | LR: {lr:.5f}")

        if composite < best_composite:
            best_composite = composite
            best_epoch = epoch
            best_ade = mean_ade
            best_per_h = per_h_ade[:]
            best_bce = bce
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            config = {"input_dim": 8, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2}
            with open("model_config.json", "w") as f:
                json.dump(config, f)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    elapsed = time.time() - t_start
    h_str = ", ".join(f"H{i+1}: {a:.1f}" for i, a in enumerate(best_per_h))
    print(f"\nTRAINING COMPLETE ({elapsed:.0f}s)")
    print(f"Best epoch: {best_epoch}")
    print(f"Best dev composite: {best_composite:.4f}")
    print(f"Best dev ADE: {best_ade:.1f} px [{h_str}]")
    print(f"Best dev BCE: {best_bce:.4f}")
    print(f"Baseline was: 0.8311")
    print(f"Improvement: {0.8311 - best_composite:.4f}")


if __name__ == "__main__":
    main()
