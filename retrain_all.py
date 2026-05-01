"""Retrain CatBoost intent + XGB trajectory on train_all.parquet (all data including dev).
For blind final submission — no dev evaluation possible."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor

from predict import _engineered_features, _build_gru_input
from trajectory_model import CrossingModel

DATA = Path(__file__).parent / "data"
MODEL_SEEDS = [42, 123, 456]
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row):
    return {k: row[k] for k in REQUEST_FIELDS}


def featurize(df):
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
        if i % 10000 == 0:
            print(f"  intent features {i}/{n}")
    return X


def build_traj_features(df, intent_model):
    n = len(df)
    feats_list = []
    targets = np.zeros((n, 4, 2), dtype=np.float64)
    for i in range(n):
        row = df.iloc[i]
        req = row_to_request(row)
        hand = _engineered_features(req)

        hist = np.stack([np.asarray(b, dtype=np.float64) for b in req["bbox_history"]])
        cx = (hist[:, 0] + hist[:, 2]) * 0.5
        cy = (hist[:, 1] + hist[:, 3]) * 0.5
        h = hist[:, 3] - hist[:, 1]
        fw, fh = float(req["frame_w"]), float(req["frame_h"])
        vx = np.diff(cx)
        vy = np.diff(cy)

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

        ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
        depth_est = (fw * 0.7 * 1.7) / max(h[-1], 10.0)
        depth_norm = min(depth_est / 100.0, 1.0)
        ego_braking = (ego_s[-1] - ego_s[-4]) / max(ego_s.max(), 1e-6)
        lat_accel = (abs(vx[-2:].mean()) - abs(vx[-6:-4].mean() if len(vx) >= 6 else vx[:2].mean())) / fw
        vx_var = vx.var() / (fw * fw)
        vy_var = vy.var() / (fh * fh)
        heading_val = np.arctan2(vy[-4:].mean(), vx[-4:].mean())
        pos_heading = (cx[-1] / fw) * heading_val / np.pi
        extra2 = np.array([depth_norm, ego_braking, lat_accel, vx_var, vy_var, pos_heading],
                          dtype=np.float32)

        intent_prob = intent_model.predict_proba(hand.reshape(1, -1))[0, 1]
        feats_list.append(np.concatenate([hand, extra, extra2, [intent_prob]]))

        cur_cx, cur_cy = cx[-1], cy[-1]
        for h_idx, hk in enumerate(HORIZON_KEYS):
            fb = np.asarray(row[hk], dtype=np.float64)
            targets[i, h_idx, 0] = (fb[0] + fb[2]) * 0.5 - cur_cx
            targets[i, h_idx, 1] = (fb[1] + fb[3]) * 0.5 - cur_cy

        if i % 10000 == 0:
            print(f"  traj features {i}/{n}")
    return np.array(feats_list, dtype=np.float32), targets


def get_gru_predictions(df, models):
    device = next(models[0].parameters()).device
    n = len(df)
    per_seed_preds = np.zeros((len(models), n, 4, 2), dtype=np.float64)
    batch_size = 1024
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inputs_orig = []
        fws, fhs = [], []
        for i in range(start, end):
            req = row_to_request(df.iloc[i])
            inputs_orig.append(_build_gru_input(req))
            fws.append(float(req["frame_w"]))
            fhs.append(float(req["frame_h"]))

        batch_orig = torch.cat(inputs_orig, dim=0).to(device)
        batch_flip = batch_orig.clone()
        batch_flip[:, :, 0] = 1.0 - batch_flip[:, :, 0]
        batch_flip[:, :, 4] = -batch_flip[:, :, 4]
        batch_flip[:, :, 7] = -batch_flip[:, :, 7]
        batch_flip[:, :, 8] = -batch_flip[:, :, 8]
        batch_flip[:, :, 10] = -batch_flip[:, :, 10]
        batch_flip[:, :, 12] = -batch_flip[:, :, 12]

        for s, model in enumerate(models):
            with torch.no_grad():
                t_orig, _ = model(batch_orig)
                t_flip, _ = model(batch_flip)
                t_flip[:, :, 0] = -t_flip[:, :, 0]
            seed_avg = ((t_orig + t_flip) / 2.0).cpu().numpy()
            for j in range(end - start):
                per_seed_preds[s, start + j, :, 0] = seed_avg[j, :, 0] * fws[j]
                per_seed_preds[s, start + j, :, 1] = seed_avg[j, :, 1] * fhs[j]

        if (start // batch_size) % 20 == 0:
            print(f"  gru {start}/{n}")
    return per_seed_preds.mean(axis=0)


def main():
    t_start = time.time()

    all_data = pd.read_parquet(DATA / "train_all.parquet")
    print(f"Training on ALL data: {len(all_data)} samples")

    # --- Step 1: Retrain CatBoost intent ---
    print("\n=== Step 1: Retrain CatBoost intent ===")
    with open("model.pkl", "rb") as f:
        old = pickle.load(f)
    old_clf = old["intent"]

    X_all = featurize(all_data)
    y_all = all_data["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  Features: {X_all.shape}, positive rate: {y_all.mean():.3f}")

    old_params = old_clf.get_all_params() if hasattr(old_clf, "get_all_params") else {}
    cb_params = {
        "iterations": old_params.get("iterations", 2000),
        "depth": old_params.get("depth", 8),
        "learning_rate": old_params.get("learning_rate", 0.0138),
        "l2_leaf_reg": old_params.get("l2_leaf_reg", 0.2115),
        "min_data_in_leaf": old_params.get("min_data_in_leaf", 19),
        "subsample": old_params.get("subsample", 0.6177),
        "bootstrap_type": old_params.get("bootstrap_type", "MVS"),
        "boosting_type": old_params.get("boosting_type", "Plain"),
        "task_type": "GPU",
        "random_seed": 42,
        "verbose": 100,
    }
    clf = CatBoostClassifier(**cb_params)
    clf.fit(X_all, y_all)

    # Also retrain LightGBM if it was part of the ensemble
    if "lgbm" in old:
        print("\n=== Step 1b: Retrain LightGBM intent ===")
        old_lgbm = old["lgbm"]
        lgbm_params = old_lgbm.get_params()
        lgbm_params.pop("early_stopping_rounds", None)
        lgbm_params.pop("early_stopping_round", None)
        n_lgbm = lgbm_params.get("n_estimators", 1000)
        lgbm_params["n_estimators"] = n_lgbm
        lgbm_clf = LGBMClassifier(**lgbm_params)
        lgbm_clf.fit(X_all, y_all)
        lgbm_weight = old.get("lgbm_weight", 0.0)
        with open("model.pkl", "wb") as f:
            pickle.dump({"intent": clf, "lgbm": lgbm_clf, "lgbm_weight": lgbm_weight}, f)
        print(f"  Saved model.pkl (CatBoost + LightGBM retrained on all data, lgbm_weight={lgbm_weight:.2f})")
    else:
        with open("model.pkl", "wb") as f:
            pickle.dump({"intent": clf}, f)
        print("  Saved model.pkl (CatBoost retrained on all data)")

    # --- Step 2: Retrain XGB trajectory ---
    print("\n=== Step 2: Build trajectory features ===")
    X_traj, y_traj = build_traj_features(all_data, clf)
    print(f"  Feature shape: {X_traj.shape}")

    print("\n=== Step 3: Get GRU predictions ===")
    with open("model_config.json") as f:
        cfg = json.load(f)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for seed in MODEL_SEEDS:
        m = CrossingModel(**cfg)
        m.load_state_dict(torch.load(f"best_model_s{seed}.pt", map_location=DEVICE, weights_only=True))
        m.to(DEVICE)
        m.eval()
        models.append(m)

    gru_preds = get_gru_predictions(all_data, models)

    print("\n=== Step 4: Train XGB regressors ===")
    with open("traj_xgb.pkl", "rb") as f:
        old_xgb = pickle.load(f)
    old_weights = old_xgb.get("blend_weights", [0.5, 0.5, 0.5, 0.5])

    xgb_models = {}
    for h in range(4):
        for c, coord_name in enumerate(["dx", "dy"]):
            key = f"h{h}_{coord_name}"
            old_model = old_xgb["models"][key]
            params = old_model.get_params()
            params.pop("early_stopping_rounds", None)
            params.pop("n_jobs", None)
            params["device"] = "cuda"
            n_est = old_model.best_iteration + 1 if hasattr(old_model, "best_iteration") and old_model.best_iteration else params.get("n_estimators", 1000)
            params["n_estimators"] = n_est
            reg = XGBRegressor(**params)
            reg.fit(X_traj, y_traj[:, h, c], verbose=False)
            xgb_models[key] = reg
            print(f"  Trained {key} (n_est={n_est})")

    for m in xgb_models.values():
        m.set_params(device="cpu")
    with open("traj_xgb.pkl", "wb") as f:
        pickle.dump({"models": xgb_models, "blend_weights": old_weights}, f)
    print(f"\n  Saved traj_xgb.pkl (retrained on all data, {len(xgb_models)} models)")
    print(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
