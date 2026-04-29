from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from trajectory_model import CrossingModel

MODEL_PATH = Path(__file__).parent / "model.pkl"
GRU_CONFIG = Path(__file__).parent / "model_config.json"
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
MODEL_SEEDS = [42, 123, 456]

_cached_xgb = None
_cached_gru_models = None


def _load_xgb():
    global _cached_xgb
    if _cached_xgb is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_xgb = pickle.load(f)["intent"]
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

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    ego_speed = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_yaw = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    seq = np.stack([
        cx / fw,
        cy / fh,
        w / fw,
        h / fh,
        dx / fw,
        dy / fh,
        np.clip(ego_speed, 0.0, 20.0) / 20.0,
        np.clip(ego_yaw, -10.0, 10.0) / 10.0,
    ], axis=-1)

    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0)  # [1, 16, 8]


def predict(request: dict) -> dict:
    xgb = _load_xgb()
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    intent_prob = float(xgb.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    models = _load_gru_models()
    input_orig = _build_gru_input(request)

    input_flip = input_orig.clone()
    input_flip[0, :, 0] = 1.0 - input_flip[0, :, 0]
    input_flip[0, :, 4] = -input_flip[0, :, 4]
    input_flip[0, :, 7] = -input_flip[0, :, 7]

    all_traj_preds = []
    for model in models:
        with torch.no_grad():
            traj_orig, _ = model(input_orig)
            all_traj_preds.append(traj_orig)

            traj_flip, _ = model(input_flip)
            traj_flip[:, :, 0] = -traj_flip[:, :, 0]
            all_traj_preds.append(traj_flip)

    traj_avg = torch.stack(all_traj_preds).mean(dim=0)
    traj_delta = traj_avg.squeeze(0).numpy()  # [4, 2]

    hist = _as_2d(request["bbox_history"])
    cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
    cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5
    cur_w = hist[-1, 2] - hist[-1, 0]
    cur_h = hist[-1, 3] - hist[-1, 1]
    fw = float(request["frame_w"])
    fh = float(request["frame_h"])

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
