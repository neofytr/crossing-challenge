from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from trajectory_model import CrossingModel

MODEL_PATH = Path(__file__).parent / "model.pkl"
GRU_CONFIG = Path(__file__).parent / "model_config.json"
TRAJ_XGB_PATH = Path(__file__).parent / "traj_xgb.pkl"
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
MODEL_SEEDS = [42, 123, 456]

_cached_xgb = None
_cached_gru_models = None
_cached_traj_xgb = None


def _load_xgb():
    global _cached_xgb
    if _cached_xgb is None:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and data.get("stacked"):
            _cached_xgb = data
        else:
            _cached_xgb = {"intent": data["intent"], "stacked": False}
    return _cached_xgb


def _load_gru_models():
    global _cached_gru_models
    if _cached_gru_models is not None:
        return _cached_gru_models

    with open(GRU_CONFIG) as f:
        cfg = json.load(f)

    models = []
    for seed in MODEL_SEEDS:
        model = CrossingModel(**cfg)
        path = Path(__file__).parent / f"best_model_s{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)

    _cached_gru_models = models
    return models


def _load_traj_xgb():
    global _cached_traj_xgb
    if _cached_traj_xgb is None and TRAJ_XGB_PATH.exists():
        with open(TRAJ_XGB_PATH, "rb") as f:
            _cached_traj_xgb = pickle.load(f)
    return _cached_traj_xgb


def _traj_xgb_features(req: dict) -> np.ndarray:
    hand = _engineered_features(req)
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    fw, fh = float(req["frame_w"]), float(req["frame_h"])
    t = np.arange(4)
    px = np.polyfit(t, cx[-4:], 1)
    py = np.polyfit(t, cy[-4:], 1)
    extra = np.array([
        px[0] / fw, py[0] / fh, px[1] / fw, py[1] / fh,
        float(cx[-1] - cx[-4]) / fw,
        float(cy[-1] - cy[-4]) / fh,
        float(cx[-1] - cx[-8]) / fw if len(cx) >= 8 else 0.0,
        float(cy[-1] - cy[-8]) / fh if len(cy) >= 8 else 0.0,
    ], dtype=np.float32)
    return np.concatenate([hand, extra]).reshape(1, -1)


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


def predict(request: dict) -> dict:
    xgb_data = _load_xgb()
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

    models = _load_gru_models()
    input_orig = _build_gru_input(request)

    if xgb_data.get("stacked"):
        clf = xgb_data["intent"]
        enc_list = []
        for model in models:
            with torch.no_grad():
                enc_list.append(model.encode(input_orig))
        gru_feats = torch.stack(enc_list).mean(dim=0).numpy()
        stacked_feats = np.concatenate([feats, gru_feats], axis=1)
        intent_prob = float(clf.predict_proba(stacked_feats)[0, 1])
    else:
        clf = xgb_data["intent"]
        intent_prob = float(clf.predict_proba(feats)[0, 1])

    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    input_flip = input_orig.clone()
    input_flip[0, :, 0] = 1.0 - input_flip[0, :, 0]
    input_flip[0, :, 4] = -input_flip[0, :, 4]
    input_flip[0, :, 7] = -input_flip[0, :, 7]
    input_flip[0, :, 8] = -input_flip[0, :, 8]
    input_flip[0, :, 10] = -input_flip[0, :, 10]
    input_flip[0, :, 12] = -input_flip[0, :, 12]

    per_seed_traj = []
    all_intent_preds = []
    for model in models:
        with torch.no_grad():
            traj_orig, intent_orig = model(input_orig)
            traj_flip, intent_flip = model(input_flip)
            traj_flip[:, :, 0] = -traj_flip[:, :, 0]
        per_seed_traj.append((traj_orig + traj_flip) / 2.0)
        all_intent_preds.append(intent_orig.item())
        all_intent_preds.append(intent_flip.item())

    traj_avg = torch.stack(per_seed_traj).mean(dim=0)
    traj_delta = traj_avg.squeeze(0).numpy()

    gru_intent_prob = float(np.mean(all_intent_preds))
    intent_prob = 0.80 * intent_prob + 0.20 * gru_intent_prob

    hist = _as_2d(request["bbox_history"])
    cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
    cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5
    cur_w = hist[-1, 2] - hist[-1, 0]
    cur_h = hist[-1, 3] - hist[-1, 1]
    fw = float(request["frame_w"])
    fh = float(request["frame_h"])

    traj_xgb_data = _load_traj_xgb()
    if traj_xgb_data is not None and traj_xgb_data.get("meta_learner"):
        traj_feats = _traj_xgb_features(request)
        if not np.isfinite(traj_feats).all():
            traj_feats = np.nan_to_num(traj_feats, nan=0.0, posinf=0.0, neginf=0.0)
        per_seed_px = np.zeros((len(models), 4, 2))
        for s, st in enumerate(per_seed_traj):
            st_np = st.squeeze(0).numpy()
            per_seed_px[s, :, 0] = st_np[:, 0] * fw
            per_seed_px[s, :, 1] = st_np[:, 1] * fh
        gru_avg_px = per_seed_px.mean(axis=0)
        gru_flat = gru_avg_px.reshape(1, -1)
        gru_std = per_seed_px.std(axis=0).reshape(1, -1)
        gru_mag = np.sqrt(gru_avg_px[:, 0]**2 + gru_avg_px[:, 1]**2).reshape(1, -1)
        meta_feats = np.concatenate([traj_feats, gru_flat, gru_std, gru_mag], axis=1)
        if not np.isfinite(meta_feats).all():
            meta_feats = np.nan_to_num(meta_feats, nan=0.0, posinf=0.0, neginf=0.0)
        xgb_models = traj_xgb_data["models"]
        for h in range(4):
            traj_delta[h, 0] = float(xgb_models[f"h{h}_dx"].predict(meta_feats)[0]) / fw
            traj_delta[h, 1] = float(xgb_models[f"h{h}_dy"].predict(meta_feats)[0]) / fh
    elif traj_xgb_data is not None:
        traj_feats = _traj_xgb_features(request)
        if not np.isfinite(traj_feats).all():
            traj_feats = np.nan_to_num(traj_feats, nan=0.0, posinf=0.0, neginf=0.0)
        xgb_models = traj_xgb_data["models"]
        blend_w = traj_xgb_data["blend_weights"]
        for h in range(4):
            xgb_dx = float(xgb_models[f"h{h}_dx"].predict(traj_feats)[0])
            xgb_dy = float(xgb_models[f"h{h}_dy"].predict(traj_feats)[0])
            gru_dx = traj_delta[h, 0] * fw
            gru_dy = traj_delta[h, 1] * fh
            traj_delta[h, 0] = (blend_w[h] * xgb_dx + (1 - blend_w[h]) * gru_dx) / fw
            traj_delta[h, 1] = (blend_w[h] * xgb_dy + (1 - blend_w[h]) * gru_dy) / fh

    out: dict[str, object] = {"intent": intent_prob}
    for i, key in enumerate(HORIZON_KEYS):
        pred_cx = cur_cx + traj_delta[i, 0] * fw
        pred_cy = cur_cy + traj_delta[i, 1] * fh
        bbox = [
            pred_cx - cur_w / 2, pred_cy - cur_h / 2,
            pred_cx + cur_w / 2, pred_cy + cur_h / 2,
        ]
        out[key] = [float(v) if np.isfinite(v) else 0.0 for v in bbox]
    return out
