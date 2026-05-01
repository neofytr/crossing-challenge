from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

from predict import _engineered_features, _build_gru_input
from trajectory_model import CrossingModel

with open("model.pkl", "rb") as f:
    _intent_data = pickle.load(f)
_intent_model = _intent_data["intent"] if isinstance(_intent_data, dict) else _intent_data

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


def build_traj_features(df):
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

        intent_prob = _intent_model.predict_proba(hand.reshape(1, -1))[0, 1]
        feats_list.append(np.concatenate([hand, extra, [intent_prob]]))

        cur_cx, cur_cy = cx[-1], cy[-1]
        for h_idx, hk in enumerate(HORIZON_KEYS):
            fb = np.asarray(row[hk], dtype=np.float64)
            targets[i, h_idx, 0] = (fb[0] + fb[2]) * 0.5 - cur_cx
            targets[i, h_idx, 1] = (fb[1] + fb[3]) * 0.5 - cur_cy

        if i % 10000 == 0:
            print(f"  {i}/{n}")
    return np.array(feats_list, dtype=np.float32), targets


def get_gru_predictions(df, models):
    n = len(df)
    num_seeds = len(models)
    per_seed_preds = np.zeros((num_seeds, n, 4, 2), dtype=np.float64)
    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inputs_orig = []
        fws, fhs = [], []
        for i in range(start, end):
            req = row_to_request(df.iloc[i])
            inputs_orig.append(_build_gru_input(req))
            fws.append(float(req["frame_w"]))
            fhs.append(float(req["frame_h"]))

        batch_orig = torch.cat(inputs_orig, dim=0)
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
            seed_avg = ((t_orig + t_flip) / 2.0).numpy()
            for j in range(end - start):
                per_seed_preds[s, start + j, :, 0] = seed_avg[j, :, 0] * fws[j]
                per_seed_preds[s, start + j, :, 1] = seed_avg[j, :, 1] * fhs[j]

        if (start // batch_size) % 20 == 0:
            print(f"  {start}/{n}")

    all_preds = per_seed_preds.mean(axis=0)
    return all_preds, per_seed_preds


def build_meta_features(X_hand, gru_preds, gru_per_seed):
    n = X_hand.shape[0]
    gru_flat = gru_preds.reshape(n, -1).astype(np.float32)
    gru_std = gru_per_seed.std(axis=0).reshape(n, -1).astype(np.float32)
    gru_mag = np.sqrt(gru_preds[:, :, 0]**2 + gru_preds[:, :, 1]**2).astype(np.float32)
    return np.concatenate([X_hand, gru_flat, gru_std, gru_mag], axis=1)


def main():
    print("Loading data...")
    train_path = DATA / "train_full.parquet"
    if not train_path.exists():
        train_path = DATA / "train.parquet"
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(DATA / "dev.parquet")

    print("Building trajectory features...")
    t0 = time.time()
    X_train, y_train = build_traj_features(train)
    X_dev, y_dev = build_traj_features(dev)
    print(f"  {time.time()-t0:.1f}s  hand-crafted shape: {X_train.shape}")

    print("Loading GRU models...")
    with open("model_config.json") as f:
        cfg = json.load(f)
    models = []
    for seed in MODEL_SEEDS:
        model = CrossingModel(**cfg)
        model.load_state_dict(torch.load(f"best_model_s{seed}.pt", map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)

    print("Getting GRU predictions (per-seed)...")
    t0 = time.time()
    gru_preds_train, gru_per_seed_train = get_gru_predictions(train, models)
    gru_preds_dev, gru_per_seed_dev = get_gru_predictions(dev, models)
    print(f"  {time.time()-t0:.1f}s")

    gru_ade_per_h = []
    for h in range(4):
        ade_h = np.sqrt((gru_preds_dev[:, h, 0] - y_dev[:, h, 0])**2 +
                        (gru_preds_dev[:, h, 1] - y_dev[:, h, 1])**2).mean()
        gru_ade_per_h.append(ade_h)
    gru_mean_ade = np.mean(gru_ade_per_h)
    print(f"\nGRU-only ADE: {gru_mean_ade:.1f}px  [{' '.join(f'H{i+1}:{a:.1f}' for i,a in enumerate(gru_ade_per_h))}]")

    # === Approach 1: XGBoost on hand-crafted features + blend ===
    print("\n--- Approach 1: XGBoost blend (hand-crafted features) ---")
    print("Training 8 XGBoost regressors...")
    xgb_models_blend = {}
    xgb_preds_dev = np.zeros_like(y_dev)
    for h in range(4):
        for c, coord_name in enumerate(["dx", "dy"]):
            key = f"h{h}_{coord_name}"
            reg = XGBRegressor(
                n_estimators=2000,
                max_depth=7,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=5,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mae",
                early_stopping_rounds=50,
            )
            reg.fit(X_train, y_train[:, h, c],
                    eval_set=[(X_dev, y_dev[:, h, c])], verbose=False)
            xgb_models_blend[key] = reg
            xgb_preds_dev[:, h, c] = reg.predict(X_dev)

    xgb_ade_per_h = []
    for h in range(4):
        ade_h = np.sqrt((xgb_preds_dev[:, h, 0] - y_dev[:, h, 0])**2 +
                        (xgb_preds_dev[:, h, 1] - y_dev[:, h, 1])**2).mean()
        xgb_ade_per_h.append(ade_h)
    xgb_mean_ade = np.mean(xgb_ade_per_h)
    print(f"XGB-only ADE: {xgb_mean_ade:.1f}px  [{' '.join(f'H{i+1}:{a:.1f}' for i,a in enumerate(xgb_ade_per_h))}]")

    print("Finding optimal blend weights per horizon...")
    best_weights = []
    for h in range(4):
        best_ade = float("inf")
        best_w = 0.0
        for w in np.arange(0.0, 1.01, 0.01):
            blended = w * xgb_preds_dev[:, h, :] + (1 - w) * gru_preds_dev[:, h, :]
            ade = np.sqrt((blended[:, 0] - y_dev[:, h, 0])**2 +
                          (blended[:, 1] - y_dev[:, h, 1])**2).mean()
            if ade < best_ade:
                best_ade = ade
                best_w = w
        best_weights.append(best_w)
        print(f"  H{h+1}: xgb_weight={best_w:.2f}, blended={best_ade:.1f}px")

    blended_ade_per_h = []
    for h in range(4):
        blended = best_weights[h] * xgb_preds_dev[:, h, :] + (1 - best_weights[h]) * gru_preds_dev[:, h, :]
        ade_h = np.sqrt((blended[:, 0] - y_dev[:, h, 0])**2 +
                        (blended[:, 1] - y_dev[:, h, 1])**2).mean()
        blended_ade_per_h.append(ade_h)
    blended_mean_ade = np.mean(blended_ade_per_h)
    print(f"Blended ADE: {blended_mean_ade:.1f}px")

    # === Approach 2: Meta-learner (hand-crafted + GRU features) ===
    print("\n--- Approach 2: XGBoost meta-learner (hand-crafted + GRU features) ---")
    X_meta_train = build_meta_features(X_train, gru_preds_train, gru_per_seed_train)
    X_meta_dev = build_meta_features(X_dev, gru_preds_dev, gru_per_seed_dev)
    print(f"Meta-feature shape: {X_meta_train.shape}")

    print("Training 8 XGBoost meta-learner regressors...")
    xgb_models_meta = {}
    meta_preds_dev = np.zeros_like(y_dev)
    for h in range(4):
        for c, coord_name in enumerate(["dx", "dy"]):
            key = f"h{h}_{coord_name}"
            reg = XGBRegressor(
                n_estimators=2000,
                max_depth=7,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=5,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mae",
                early_stopping_rounds=50,
            )
            reg.fit(X_meta_train, y_train[:, h, c],
                    eval_set=[(X_meta_dev, y_dev[:, h, c])], verbose=False)
            xgb_models_meta[key] = reg
            meta_preds_dev[:, h, c] = reg.predict(X_meta_dev)

    meta_ade_per_h = []
    for h in range(4):
        ade_h = np.sqrt((meta_preds_dev[:, h, 0] - y_dev[:, h, 0])**2 +
                        (meta_preds_dev[:, h, 1] - y_dev[:, h, 1])**2).mean()
        meta_ade_per_h.append(ade_h)
    meta_mean_ade = np.mean(meta_ade_per_h)
    print(f"Meta-learner ADE: {meta_mean_ade:.1f}px  [{' '.join(f'H{i+1}:{a:.1f}' for i,a in enumerate(meta_ade_per_h))}]")

    # === Summary and save best ===
    print(f"\n{'='*50}")
    print(f"  GRU-only mean ADE:      {gru_mean_ade:.1f}px")
    print(f"  XGB-only mean ADE:      {xgb_mean_ade:.1f}px")
    print(f"  Blended mean ADE:       {blended_mean_ade:.1f}px")
    print(f"  Meta-learner mean ADE:  {meta_mean_ade:.1f}px")

    if meta_mean_ade < blended_mean_ade:
        print(f"\nMeta-learner wins! ({meta_mean_ade:.1f} vs {blended_mean_ade:.1f})")
        with open("traj_xgb.pkl", "wb") as f:
            pickle.dump({"models": xgb_models_meta, "meta_learner": True}, f)
        print("Saved traj_xgb.pkl (meta_learner mode)")
    elif blended_mean_ade < gru_mean_ade:
        print(f"\nBlend wins! ({blended_mean_ade:.1f} vs meta {meta_mean_ade:.1f})")
        with open("traj_xgb.pkl", "wb") as f:
            pickle.dump({"models": xgb_models_blend, "blend_weights": best_weights}, f)
        print("Saved traj_xgb.pkl (blend mode)")
    else:
        print("\nNeither approach helps over GRU-only. Not saving.")


if __name__ == "__main__":
    main()
