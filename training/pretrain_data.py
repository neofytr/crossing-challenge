"""Self-supervised pretraining dataset from raw tracklets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
TRACKLETS = DATA / "tracklets_raw.parquet"
PAST_LEN = 16
FUTURE_LEN = 30
WINDOW_LEN = PAST_LEN + FUTURE_LEN  # 46
HORIZON_IDX = [8, 15, 23, 30]


class PretrainDataset(Dataset):
    def __init__(self, augment: bool = False):
        raw = pd.read_parquet(TRACKLETS)
        raw = raw[raw["frame"] % 2 == 0].copy()
        raw.sort_values(["ped_id", "frame"], inplace=True)
        raw.reset_index(drop=True, inplace=True)

        self.x1 = raw["x1"].values
        self.y1 = raw["y1"].values
        self.x2 = raw["x2"].values
        self.y2 = raw["y2"].values
        self.fw = raw["frame_w"].values
        self.fh = raw["frame_h"].values
        self.ego_speed = raw["ego_speed_ms"].fillna(0.0).values
        self.ego_yaw = raw["ego_yaw_rate"].fillna(0.0).values
        self.occlusion = raw["occlusion"].values

        self.windows = []
        frames = raw["frame"].values
        pids = raw["ped_id"].values

        current_pid = None
        run_start = 0
        for i in range(len(raw)):
            if pids[i] != current_pid:
                if current_pid is not None:
                    self._add_runs(run_start, i, frames)
                current_pid = pids[i]
                run_start = i
        if current_pid is not None:
            self._add_runs(run_start, len(raw), frames)

        self.augment = augment
        print(f"PretrainDataset: {len(self.windows)} windows from {len(raw)} frames")

    def _add_runs(self, ped_start, ped_end, frames):
        f = frames[ped_start:ped_end]
        if len(f) < WINDOW_LEN:
            return
        diffs = np.diff(f)
        breaks = np.where(diffs != 2)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(f)]))
        for s, e in zip(starts, ends):
            run_len = e - s
            if run_len < WINDOW_LEN:
                continue
            for offset in range(run_len - WINDOW_LEN + 1):
                global_idx = ped_start + s + offset
                current_idx = global_idx + PAST_LEN - 1
                if self.occlusion[current_idx] == "full":
                    continue
                self.windows.append(global_idx)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        past_end = start + PAST_LEN

        x1 = self.x1[start:past_end].copy()
        y1 = self.y1[start:past_end].copy()
        x2 = self.x2[start:past_end].copy()
        y2 = self.y2[start:past_end].copy()
        fw = float(self.fw[start])
        fh = float(self.fh[start])

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1

        dx = np.zeros(PAST_LEN, dtype=np.float64)
        dy = np.zeros(PAST_LEN, dtype=np.float64)
        dx[1:] = np.diff(cx)
        dy[1:] = np.diff(cy)

        ax = np.zeros(PAST_LEN, dtype=np.float64)
        ay = np.zeros(PAST_LEN, dtype=np.float64)
        ax[2:] = np.diff(dx[1:])
        ay[2:] = np.diff(dy[1:])

        ego_speed = self.ego_speed[start:past_end].copy()
        ego_yaw = self.ego_yaw[start:past_end].copy()

        f_est = fw * 0.7
        dt = 1.0 / 15.0
        ego_yaw_c = np.clip(ego_yaw, -0.5, 0.5)
        ego_speed_c = np.clip(ego_speed, 0.0, 15.0)
        ego_dx = np.zeros(PAST_LEN, dtype=np.float64)
        ego_dy = np.zeros(PAST_LEN, dtype=np.float64)
        for t in range(PAST_LEN):
            depth_t = f_est * 1.7 / max(h[t], 10.0)
            ego_dx[t] = f_est * ego_yaw_c[t] * dt
            cx_rel = cx[t] - fw / 2.0
            cy_rel = cy[t] - fh / 2.0
            ego_dx[t] += cx_rel * ego_speed_c[t] * dt / depth_t
            ego_dy[t] += cy_rel * ego_speed_c[t] * dt / depth_t
        ego_dx = np.clip(ego_dx, -50.0, 50.0)
        ego_dy = np.clip(ego_dy, -50.0, 50.0)

        dx_comp = dx - ego_dx
        dy_comp = dy - ego_dy
        ax_comp = np.zeros(PAST_LEN, dtype=np.float64)
        ay_comp = np.zeros(PAST_LEN, dtype=np.float64)
        ax_comp[2:] = np.diff(dx_comp[1:])
        ay_comp[2:] = np.diff(dy_comp[1:])

        cur_cx = float(cx[-1])
        cur_cy = float(cy[-1])
        targets = np.zeros((4, 2), dtype=np.float64)
        for i, h_idx in enumerate(HORIZON_IDX):
            fut_idx = start + PAST_LEN + h_idx - 1
            fut_cx = (self.x1[fut_idx] + self.x2[fut_idx]) * 0.5
            fut_cy = (self.y1[fut_idx] + self.y2[fut_idx]) * 0.5
            targets[i, 0] = (fut_cx - cur_cx) / fw
            targets[i, 1] = (fut_cy - cur_cy) / fh

        if self.augment:
            if torch.rand(1).item() < 0.5:
                cx = fw - cx
                dx = -dx; ax = -ax
                dx_comp = -dx_comp; ax_comp = -ax_comp
                ego_dx = -ego_dx; ego_yaw = -ego_yaw
                targets[:, 0] = -targets[:, 0]
            if torch.rand(1).item() < 0.3:
                speed_scale = 0.85 + torch.rand(1).item() * 0.30
                dx *= speed_scale; dy *= speed_scale
                ax *= speed_scale; ay *= speed_scale
                dx_comp *= speed_scale; dy_comp *= speed_scale
                ax_comp *= speed_scale; ay_comp *= speed_scale
                targets *= speed_scale

        seq = np.stack([
            cx / fw, cy / fh, w / fw, h / fh,
            dx_comp / fw, dy_comp / fh,
            np.clip(ego_speed, 0.0, 20.0) / 20.0,
            np.clip(ego_yaw, -10.0, 10.0) / 10.0,
            ax_comp / fw, ay_comp / fh,
            dx / fw, dy / fh,
            ego_dx / fw, ego_dy / fh,
        ], axis=-1)

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(targets),
            torch.tensor([fw, fh], dtype=torch.float32),
            torch.tensor([cur_cx, cur_cy], dtype=torch.float32),
        )
