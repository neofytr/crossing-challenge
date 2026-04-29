from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA = Path(__file__).parent / "data"

HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


class TrajectoryDataset(Dataset):
    def __init__(self, parquet_path: str | Path, augment: bool = False):
        self.df = pd.read_parquet(parquet_path)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fw = float(row["frame_w"])
        fh = float(row["frame_h"])

        hist = np.stack([np.asarray(b, dtype=np.float64) for b in row["bbox_history"]])  # (16, 4)

        cx = (hist[:, 0] + hist[:, 2]) * 0.5
        cy = (hist[:, 1] + hist[:, 3]) * 0.5
        w = hist[:, 2] - hist[:, 0]
        h = hist[:, 3] - hist[:, 1]

        dx = np.zeros(16, dtype=np.float64)
        dy = np.zeros(16, dtype=np.float64)
        dx[1:] = np.diff(cx)
        dy[1:] = np.diff(cy)

        ego_speed = np.asarray(row["ego_speed_history"], dtype=np.float64)
        ego_yaw = np.asarray(row["ego_yaw_history"], dtype=np.float64)

        cur_cx = float(cx[-1])
        cur_cy = float(cy[-1])
        cur_w = float(w[-1])
        cur_h = float(h[-1])

        # Future target displacements (normalized)
        targets = np.zeros((4, 2), dtype=np.float64)
        for i, hk in enumerate(HORIZON_KEYS):
            fb = np.asarray(row[hk], dtype=np.float64)
            fcx = (fb[0] + fb[2]) * 0.5
            fcy = (fb[1] + fb[3]) * 0.5
            targets[i, 0] = (fcx - cur_cx) / fw
            targets[i, 1] = (fcy - cur_cy) / fh

        intent = float(row["will_cross_2s"])

        ax = np.zeros(16, dtype=np.float64)
        ay = np.zeros(16, dtype=np.float64)
        ax[2:] = np.diff(dx[1:])
        ay[2:] = np.diff(dy[1:])

        if self.augment:
            if torch.rand(1).item() < 0.5:
                cx = fw - cx
                dx = -dx
                ax = -ax
                ego_yaw = -ego_yaw
                cur_cx = fw - cur_cx
                targets[:, 0] = -targets[:, 0]
            if torch.rand(1).item() < 0.3:
                speed_scale = 0.85 + torch.rand(1).item() * 0.30
                dx = dx * speed_scale
                dy = dy * speed_scale
                ax = ax * speed_scale
                ay = ay * speed_scale
                targets = targets * speed_scale

        seq = np.stack([
            cx / fw,
            cy / fh,
            w / fw,
            h / fh,
            dx / fw,
            dy / fh,
            np.clip(ego_speed, 0.0, 20.0) / 20.0,
            np.clip(ego_yaw, -10.0, 10.0) / 10.0,
            ax / fw,
            ay / fh,
        ], axis=-1)  # (16, 10)

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return (
            torch.from_numpy(seq),                                          # [16, 8]
            torch.from_numpy(targets),                                      # [4, 2]
            torch.tensor([intent], dtype=torch.float32),                    # [1]
            torch.tensor([fw, fh], dtype=torch.float32),                    # [2]
            torch.tensor([cur_cx, cur_cy], dtype=torch.float32),            # [2]
            torch.tensor([cur_w, cur_h], dtype=torch.float32),              # [2]
        )


def build_dataloaders(batch_size: int = 256, worker_init_fn=None, generator=None):
    train_ds = TrajectoryDataset(DATA / "train.parquet", augment=True)
    dev_ds = TrajectoryDataset(DATA / "dev.parquet", augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=2, drop_last=True,
                          worker_init_fn=worker_init_fn, generator=generator)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=2)
    return train_dl, dev_dl
