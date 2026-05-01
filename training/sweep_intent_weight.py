"""Sweep CatBoost/GRU intent ensemble weight to find optimal BCE."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---- import model class (suppresses param count print) ----
import io
from contextlib import redirect_stdout

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

with redirect_stdout(io.StringIO()):
    from trajectory_model import CrossingModel

# ---- paths ----
MODEL_PATH = ROOT / "model.pkl"
GRU_CONFIG = ROOT / "model_config.json"
DEV_PATH = ROOT / "data" / "dev.parquet"
MODEL_SEEDS = [42, 123, 456]

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]

BCE_FLOOR = 0.2488
ADE_FLOOR = 49.80


# ---- copied from predict.py (verbatim, no modifications) ----

def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _engineered_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)
    vy = np.diff(cy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    feats = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx.std() / fw,
        vy.std() / fh,
        (h / (w + 1e-6)).mean(),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]

    diag = np.sqrt(fw**2 + fh**2)
    total_disp = np.sqrt((cx[-1] - cx[0])**2 + (cy[-1] - cy[0])**2) / diag

    vx_recent = vx[-4:].mean()
    vy_recent = vy[-4:].mean()
    vel_magnitude = np.sqrt(vx_recent**2 + vy_recent**2) / diag
    heading = np.arctan2(vy_recent, vx_recent) / np.pi

    if len(vx) >= 5:
        ax_recent = np.diff(vx[-5:]).mean() / fw
        ay_recent = np.diff(vy[-5:]).mean() / fh
    else:
        ax_recent = 0.0
        ay_recent = 0.0
    accel_magnitude = np.sqrt(ax_recent**2 + ay_recent**2)

    w_change_rate = (w[-1] - w[0]) / (15.0 * fw)
    h_change_rate = (h[-1] - h[0]) / (15.0 * fh)
    area_ratio = (w[-1] * h[-1]) / (w[0] * h[0] + 1e-6)
    lower_half = 1.0 if cy[-1] > fh / 2 else 0.0

    feats.extend([
        total_disp,
        vel_magnitude,
        heading,
        ax_recent,
        ay_recent,
        accel_magnitude,
        w_change_rate,
        h_change_rate,
        area_ratio,
        lower_half,
    ])

    vx_to_vy_ratio = np.abs(vx[-4:].mean()) / (np.abs(vy[-4:].mean()) + 1e-6)
    vx_2 = vx[-2:].mean() / fw if len(vx) >= 2 else 0.0
    vy_2 = vy[-2:].mean() / fh if len(vy) >= 2 else 0.0
    vx_8 = vx[-8:].mean() / fw if len(vx) >= 8 else vx.mean() / fw
    vy_8 = vy[-8:].mean() / fh if len(vy) >= 8 else vy.mean() / fh
    vx_trend = (vx[-2:].mean() - vx[-8:].mean()) / fw if len(vx) >= 8 else 0.0
    proximity_center_x = abs(cx[-1] / fw - 0.5)
    ego_speed_change = ego_s[-1] - ego_s[0]
    ego_ped_interaction = ego_s[-1] * abs(vx[-4:].mean()) / fw
    bbox_area = (w[-1] * h[-1]) / (fw * fh)
    ar_current = h[-1] / (w[-1] + 1e-6)
    ar_first = h[0] / (w[0] + 1e-6)
    aspect_ratio_change = ar_current - ar_first
    max_lateral_speed = np.abs(vx).max() / fw
    lateral_disp = abs(cx[-1] - cx[0]) / fw
    longitudinal_disp = abs(cy[-1] - cy[0]) / fh
    lat_to_long_disp_ratio = lateral_disp / (longitudinal_disp + 1e-6)
    speed_per_frame = np.sqrt(np.concatenate([[0], vx])**2 + np.concatenate([[0], vy])**2)
    stationary_frames = float(np.sum(speed_per_frame < 1.0)) / 16.0
    vx_sign_changes = float(np.sum(np.diff(np.sign(vx)) != 0)) / 14.0

    feats.extend([
        min(vx_to_vy_ratio, 10.0),
        vx_2, vy_2, vx_8, vy_8,
        vx_trend,
        proximity_center_x,
        ego_speed_change,
        ego_ped_interaction,
        bbox_area,
        aspect_ratio_change,
        max_lateral_speed,
        lateral_disp,
        longitudinal_disp,
        min(lat_to_long_disp_ratio, 10.0),
        stationary_frames,
        vx_sign_changes,
    ])

    f_est = fw * 0.7
    dt_frame = 1.0 / 15.0
    ego_y_c = np.clip(ego_y, -0.5, 0.5)
    ego_s_c = np.clip(ego_s, 0.0, 15.0)
    ego_dx_arr = np.zeros(15, dtype=np.float64)
    ego_dy_arr = np.zeros(15, dtype=np.float64)
    for t in range(15):
        t_idx = t + 1
        depth_t = f_est * 1.7 / max(h[t_idx], 10.0)
        ego_dx_arr[t] = f_est * ego_y_c[t_idx] * dt_frame
        cx_rel = cx[t_idx] - fw / 2.0
        cy_rel = cy[t_idx] - fh / 2.0
        ego_dx_arr[t] += cx_rel * ego_s_c[t_idx] * dt_frame / depth_t
        ego_dy_arr[t] += cy_rel * ego_s_c[t_idx] * dt_frame / depth_t
    ego_dx_arr = np.clip(ego_dx_arr, -50.0, 50.0)
    ego_dy_arr = np.clip(ego_dy_arr, -50.0, 50.0)

    vx_comp = vx - ego_dx_arr
    vy_comp = vy - ego_dy_arr

    vx_comp_recent = vx_comp[-4:].mean() / fw
    vy_comp_recent = vy_comp[-4:].mean() / fh
    vx_comp_heading = np.arctan2(vy_comp[-4:].mean(), vx_comp[-4:].mean()) / np.pi
    lateral_comp = abs(vx_comp[-4:].mean()) / fw

    feats.extend([
        vx_comp_recent,
        vy_comp_recent,
        min(abs(vx_comp_recent) / (abs(vy_comp_recent) + 1e-6), 10.0),
        vx_comp_heading,
        lateral_comp,
    ])
    return np.asarray(feats, dtype=np.float32)


def _build_gru_input(req: dict) -> torch.Tensor:
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]

    dx = np.zeros(16, dtype=np.float64)
    dy = np.zeros(16, dtype=np.float64)
    dx[1:] = np.diff(cx)
    dy[1:] = np.diff(cy)

    ax = np.zeros(16, dtype=np.float64)
    ay = np.zeros(16, dtype=np.float64)
    ax[2:] = np.diff(dx[1:])
    ay[2:] = np.diff(dy[1:])

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    ego_speed = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_yaw = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    f_est = fw * 0.7
    dt = 1.0 / 15.0
    ego_yaw_c = np.clip(ego_yaw, -0.5, 0.5)
    ego_speed_c = np.clip(ego_speed, 0.0, 15.0)
    ego_dx = np.zeros(16, dtype=np.float64)
    ego_dy = np.zeros(16, dtype=np.float64)
    for t in range(16):
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
    ax_comp = np.zeros(16, dtype=np.float64)
    ay_comp = np.zeros(16, dtype=np.float64)
    ax_comp[2:] = np.diff(dx_comp[1:])
    ay_comp[2:] = np.diff(dy_comp[1:])

    seq = np.stack([
        cx / fw,
        cy / fh,
        w / fw,
        h / fh,
        dx_comp / fw,
        dy_comp / fh,
        np.clip(ego_speed, 0.0, 20.0) / 20.0,
        np.clip(ego_yaw, -10.0, 10.0) / 10.0,
        ax_comp / fw,
        ay_comp / fh,
        dx / fw,
        dy / fh,
        ego_dx / fw,
        ego_dy / fh,
    ], axis=-1)

    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0)  # [1, 16, 14]


# ---- main sweep ----

def main():
    t0 = time.time()

    # Load dev data
    print("Loading dev data...", flush=True)
    df = pd.read_parquet(DEV_PATH)
    records = df[REQUEST_FIELDS].to_dict("records")
    targets = df["will_cross_2s"].to_numpy(dtype=np.float64)
    n = len(records)
    print(f"  {n} rows, positive rate {targets.mean():.4f}")

    # Load CatBoost model
    print("Loading CatBoost model...", flush=True)
    with open(MODEL_PATH, "rb") as f:
        xgb_data = pickle.load(f)
    if isinstance(xgb_data, dict) and xgb_data.get("stacked"):
        use_stacked = True
        clf = xgb_data["intent"]
        print("  Using stacked model (CatBoost + GRU encoder features)")
    else:
        use_stacked = False
        clf = xgb_data["intent"]
        print("  Using plain CatBoost model")

    # Load GRU models
    print("Loading GRU models...", flush=True)
    with open(GRU_CONFIG) as f:
        cfg = json.load(f)
    gru_models = []
    for seed in MODEL_SEEDS:
        with redirect_stdout(io.StringIO()):
            model = CrossingModel(**cfg)
        path = ROOT / f"best_model_s{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        gru_models.append(model)
    print(f"  Loaded {len(gru_models)} GRU models")

    # Compute per-row probabilities
    print(f"Computing probabilities for {n} rows...", flush=True)
    catboost_probs = np.zeros(n, dtype=np.float64)
    gru_probs = np.zeros(n, dtype=np.float64)

    for i, req in enumerate(records):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  row {i+1}/{n}  ({elapsed:.1f}s)", flush=True)

        # CatBoost features
        feats = _engineered_features(req).reshape(1, -1)
        if not np.isfinite(feats).all():
            feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

        # GRU input
        input_orig = _build_gru_input(req)

        # CatBoost prediction (may use GRU encoder for stacked model)
        if use_stacked:
            enc_list = []
            for model in gru_models:
                with torch.no_grad():
                    enc_list.append(model.encode(input_orig))
            gru_feats = torch.stack(enc_list).mean(dim=0).numpy()
            stacked_feats = np.concatenate([feats, gru_feats], axis=1)
            cb_prob = float(clf.predict_proba(stacked_feats)[0, 1])
        else:
            cb_prob = float(clf.predict_proba(feats)[0, 1])

        if not np.isfinite(cb_prob):
            cb_prob = 0.5
        catboost_probs[i] = cb_prob

        # GRU intent prediction (TTA: orig + flip, 3 seeds = 6 predictions)
        input_flip = input_orig.clone()
        input_flip[0, :, 0] = 1.0 - input_flip[0, :, 0]
        input_flip[0, :, 4] = -input_flip[0, :, 4]
        input_flip[0, :, 7] = -input_flip[0, :, 7]
        input_flip[0, :, 8] = -input_flip[0, :, 8]
        input_flip[0, :, 10] = -input_flip[0, :, 10]
        input_flip[0, :, 12] = -input_flip[0, :, 12]

        all_intent_preds = []
        for model in gru_models:
            with torch.no_grad():
                _, intent_orig = model(input_orig)
                _, intent_flip = model(input_flip)
            all_intent_preds.append(intent_orig.item())
            all_intent_preds.append(intent_flip.item())

        gru_probs[i] = float(np.mean(all_intent_preds))

    elapsed = time.time() - t0
    print(f"  Done computing probabilities in {elapsed:.1f}s\n", flush=True)

    # Sanity checks
    print(f"CatBoost probs: mean={catboost_probs.mean():.4f}  min={catboost_probs.min():.4f}  max={catboost_probs.max():.4f}")
    print(f"GRU probs:      mean={gru_probs.mean():.4f}  min={gru_probs.min():.4f}  max={gru_probs.max():.4f}")
    print()

    # Sweep weights
    def compute_bce(probs, targets):
        p = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return -float(np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p)))

    weights = np.arange(0.0, 1.001, 0.05)
    results = []

    print(f"{'CatBoost_w':>10s}  {'GRU_w':>6s}  {'BCE':>8s}  {'intent_term':>11s}  {'composite*':>10s}")
    print("-" * 60)

    for w_cb in weights:
        w_gru = 1.0 - w_cb
        blended = w_cb * catboost_probs + w_gru * gru_probs
        bce = compute_bce(blended, targets)
        intent_term = bce / BCE_FLOOR
        # composite uses current best ADE (24.5) for the traj term
        ade_term = 24.5 / ADE_FLOOR
        composite = 0.5 * intent_term + 0.5 * ade_term
        results.append((w_cb, bce, intent_term, composite))
        marker = ""
        if abs(w_cb - 0.80) < 0.001:
            marker = "  <-- current"
        print(f"{w_cb:>10.2f}  {w_gru:>6.2f}  {bce:>8.4f}  {intent_term:>11.4f}  {composite:>10.4f}{marker}")

    # Find optimal
    print()
    best_idx = np.argmin([r[1] for r in results])
    best_w, best_bce, best_intent, best_comp = results[best_idx]
    curr_idx = [i for i, r in enumerate(results) if abs(r[0] - 0.80) < 0.001][0]
    curr_bce = results[curr_idx][1]

    print(f"OPTIMAL: CatBoost weight = {best_w:.2f}, BCE = {best_bce:.4f}")
    print(f"CURRENT: CatBoost weight = 0.80, BCE = {curr_bce:.4f}")
    print(f"DELTA:   BCE change = {best_bce - curr_bce:+.4f}")
    if best_bce < curr_bce:
        print(f"         Composite improvement: {(curr_bce - best_bce) / BCE_FLOOR / 2:.4f}")
    print()

    # Also do finer sweep around optimal
    fine_start = max(0.0, best_w - 0.10)
    fine_end = min(1.0, best_w + 0.10)
    fine_weights = np.arange(fine_start, fine_end + 0.001, 0.01)

    print(f"Fine sweep around {best_w:.2f} (step=0.01):")
    print(f"{'CatBoost_w':>10s}  {'BCE':>8s}  {'composite*':>10s}")
    print("-" * 35)

    fine_results = []
    for w_cb in fine_weights:
        blended = w_cb * catboost_probs + (1.0 - w_cb) * gru_probs
        bce = compute_bce(blended, targets)
        intent_term = bce / BCE_FLOOR
        ade_term = 24.5 / ADE_FLOOR
        composite = 0.5 * intent_term + 0.5 * ade_term
        fine_results.append((w_cb, bce, composite))
        print(f"{w_cb:>10.2f}  {bce:>8.4f}  {composite:>10.4f}")

    fine_best = min(fine_results, key=lambda r: r[1])
    print(f"\nFINE OPTIMAL: CatBoost weight = {fine_best[0]:.2f}, BCE = {fine_best[1]:.4f}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
