from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
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


def compute_intent_probs(df, clf):
    """Compute intent probabilities for all samples using the CatBoost model."""
    n = len(df)
    probs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        req = row_to_request(df.iloc[i])
        feats = _engineered_features(req).reshape(1, -1)
        if not np.isfinite(feats).all():
            feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        probs[i] = float(clf.predict_proba(feats)[0, 1])
        if i % 10000 == 0 and i > 0:
            print(f"  intent {i}/{n}")
    return probs


def build_traj_features(df, intent_probs):
    """Build trajectory features with intent probability appended."""
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

        # Append intent probability as the last feature
        feat_vec = np.concatenate([hand, extra, [intent_probs[i]]])
        feats_list.append(feat_vec)

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


def compute_blended_ade(xgb_preds, gru_preds, y_true):
    """Find optimal blend weights and return mean ADE across horizons."""
    best_weights = []
    ade_per_h = []
    for h in range(4):
        best_ade = float("inf")
        best_w = 0.0
        for w in np.arange(0.0, 1.01, 0.01):
            blended = w * xgb_preds[:, h, :] + (1 - w) * gru_preds[:, h, :]
            ade = np.sqrt((blended[:, 0] - y_true[:, h, 0])**2 +
                          (blended[:, 1] - y_true[:, h, 1])**2).mean()
            if ade < best_ade:
                best_ade = ade
                best_w = w
        best_weights.append(best_w)
        ade_per_h.append(best_ade)
    return np.mean(ade_per_h), best_weights, ade_per_h


def make_objective(X_train, y_train, X_dev, y_dev, gru_preds_dev):
    """Create the Optuna objective function with pre-loaded data."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [500, 1000, 2000, 3000]),
            "max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 7, 8, 9]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [1.0, 2.0, 5.0, 10.0]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 10]),
            "tree_method": "hist",
            "n_jobs": -1,
            "eval_metric": "mae",
            "early_stopping_rounds": 50,
        }

        # Train 8 XGB models (4 horizons x 2 coords)
        xgb_preds_dev = np.zeros_like(y_dev)
        for h in range(4):
            for c in range(2):
                reg = XGBRegressor(**params)
                reg.fit(X_train, y_train[:, h, c],
                        eval_set=[(X_dev, y_dev[:, h, c])], verbose=False)
                xgb_preds_dev[:, h, c] = reg.predict(X_dev)

        # Find optimal blend weights and compute mean ADE
        mean_ade, weights, ade_per_h = compute_blended_ade(
            xgb_preds_dev, gru_preds_dev, y_dev)

        # Store blend weights as trial user attrs for retrieval later
        for h in range(4):
            trial.set_user_attr(f"blend_w_h{h}", weights[h])
            trial.set_user_attr(f"ade_h{h}", ade_per_h[h])

        print(f"  Trial {trial.number}: ADE={mean_ade:.2f}px "
              f"[{' '.join(f'H{i+1}:{a:.1f}' for i,a in enumerate(ade_per_h))}] "
              f"weights=[{', '.join(f'{w:.2f}' for w in weights)}]")

        return mean_ade

    return objective


def main():
    t_start = time.time()

    # --- Load data ---
    print("Loading data...")
    train_path = DATA / "train_full.parquet"
    if not train_path.exists():
        train_path = DATA / "train.parquet"
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  Train: {len(train)}, Dev: {len(dev)}")

    # --- Load CatBoost intent model ---
    print("Loading CatBoost intent model...")
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    if isinstance(model_data, dict) and "intent" in model_data:
        intent_clf = model_data["intent"]
    else:
        intent_clf = model_data
    print("  Computing intent probabilities for train...")
    intent_probs_train = compute_intent_probs(train, intent_clf)
    print("  Computing intent probabilities for dev...")
    intent_probs_dev = compute_intent_probs(dev, intent_clf)

    # --- Build trajectory features (with intent) ---
    print("Building trajectory features...")
    t0 = time.time()
    X_train, y_train = build_traj_features(train, intent_probs_train)
    X_dev, y_dev = build_traj_features(dev, intent_probs_dev)
    print(f"  {time.time()-t0:.1f}s  feature shape: {X_train.shape}")

    # --- Load GRU models ---
    print("Loading GRU models...")
    with open("model_config.json") as f:
        cfg = json.load(f)
    models = []
    for seed in MODEL_SEEDS:
        model = CrossingModel(**cfg)
        model.load_state_dict(torch.load(f"best_model_s{seed}.pt",
                                         map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)

    # --- Get GRU predictions ---
    print("Getting GRU predictions...")
    t0 = time.time()
    gru_preds_train, _ = get_gru_predictions(train, models)
    gru_preds_dev, _ = get_gru_predictions(dev, models)
    print(f"  {time.time()-t0:.1f}s")

    # GRU-only baseline
    gru_ade_per_h = []
    for h in range(4):
        ade_h = np.sqrt((gru_preds_dev[:, h, 0] - y_dev[:, h, 0])**2 +
                        (gru_preds_dev[:, h, 1] - y_dev[:, h, 1])**2).mean()
        gru_ade_per_h.append(ade_h)
    gru_mean_ade = np.mean(gru_ade_per_h)
    print(f"\nGRU-only ADE: {gru_mean_ade:.1f}px "
          f"[{' '.join(f'H{i+1}:{a:.1f}' for i,a in enumerate(gru_ade_per_h))}]")

    # --- Optuna search ---
    print(f"\n{'='*60}")
    print("Starting Optuna hyperparameter search (100 trials, TPE)...")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="traj_xgb_blend",
    )

    objective_fn = make_objective(X_train, y_train, X_dev, y_dev, gru_preds_dev)
    study.optimize(objective_fn, n_trials=100, show_progress_bar=False)

    # --- Results ---
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial: #{best.number}")
    print(f"Best ADE:   {best.value:.2f}px")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"Best blend weights:")
    for h in range(4):
        w = best.user_attrs[f"blend_w_h{h}"]
        a = best.user_attrs[f"ade_h{h}"]
        print(f"  H{h+1}: weight={w:.2f}, ADE={a:.1f}px")
    print(f"{'='*60}\n")

    # --- Retrain final models with best params ---
    print("Retraining final models with best hyperparameters...")
    best_params = {
        "n_estimators": best.params["n_estimators"],
        "max_depth": best.params["max_depth"],
        "learning_rate": best.params["learning_rate"],
        "subsample": best.params["subsample"],
        "colsample_bytree": best.params["colsample_bytree"],
        "reg_alpha": best.params["reg_alpha"],
        "reg_lambda": best.params["reg_lambda"],
        "min_child_weight": best.params["min_child_weight"],
        "tree_method": "hist",
        "n_jobs": -1,
        "eval_metric": "mae",
        "early_stopping_rounds": 50,
    }

    xgb_models = {}
    xgb_preds_dev = np.zeros_like(y_dev)
    for h in range(4):
        for c, coord_name in enumerate(["dx", "dy"]):
            key = f"h{h}_{coord_name}"
            reg = XGBRegressor(**best_params)
            reg.fit(X_train, y_train[:, h, c],
                    eval_set=[(X_dev, y_dev[:, h, c])], verbose=False)
            xgb_models[key] = reg
            xgb_preds_dev[:, h, c] = reg.predict(X_dev)
            print(f"  Trained {key}")

    # Find final blend weights
    print("Finding final blend weights...")
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
        print(f"  H{h+1}: xgb_weight={best_w:.2f}, blended_ADE={best_ade:.1f}px")

    blended_ade_per_h = []
    for h in range(4):
        blended = best_weights[h] * xgb_preds_dev[:, h, :] + \
                  (1 - best_weights[h]) * gru_preds_dev[:, h, :]
        ade_h = np.sqrt((blended[:, 0] - y_dev[:, h, 0])**2 +
                        (blended[:, 1] - y_dev[:, h, 1])**2).mean()
        blended_ade_per_h.append(ade_h)
    final_ade = np.mean(blended_ade_per_h)

    print(f"\n{'='*60}")
    print(f"  GRU-only mean ADE:     {gru_mean_ade:.1f}px")
    print(f"  Final blended ADE:     {final_ade:.1f}px")
    print(f"  Improvement:           {gru_mean_ade - final_ade:.1f}px")
    print(f"{'='*60}")

    # --- Save ---
    with open("traj_xgb.pkl", "wb") as f:
        pickle.dump({"models": xgb_models, "blend_weights": best_weights}, f)
    print(f"\nSaved traj_xgb.pkl (blend mode, {len(xgb_models)} models)")
    print(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
