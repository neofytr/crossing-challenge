from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from predict import _engineered_features, _build_gru_input
from trajectory_model import CrossingModel

DATA = Path(__file__).parent / "data"
MODEL_SEEDS = [42, 123, 456, 789, 1]

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row):
    return {k: row[k] for k in REQUEST_FIELDS}


def extract_gru_features(df, models):
    all_feats = []
    batch_size = 512
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inputs = []
        for i in range(start, end):
            inputs.append(_build_gru_input(row_to_request(df.iloc[i])))
        batch = torch.cat(inputs, dim=0)
        enc_list = []
        for model in models:
            with torch.no_grad():
                enc_list.append(model.encode(batch))
        enc_avg = torch.stack(enc_list).mean(dim=0).numpy()
        all_feats.append(enc_avg)
        if (start // batch_size) % 20 == 0:
            print(f"  {start}/{n}")
    return np.concatenate(all_feats, axis=0)


def extract_hand_features(df):
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
    return X


def main():
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}  dev: {len(dev):,}")

    print("Loading GRU models...")
    with open("model_config.json") as f:
        cfg = json.load(f)
    models = []
    for seed in MODEL_SEEDS:
        model = CrossingModel(**cfg)
        path = Path(__file__).parent / f"best_model_s{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)

    print("Extracting hand features...")
    t0 = time.time()
    X_train_hand = extract_hand_features(train)
    X_dev_hand = extract_hand_features(dev)
    print(f"  {time.time()-t0:.1f}s  hand shape: {X_train_hand.shape}")

    print("Extracting GRU encoder features...")
    t0 = time.time()
    X_train_gru = extract_gru_features(train, models)
    X_dev_gru = extract_gru_features(dev, models)
    print(f"  {time.time()-t0:.1f}s  gru shape: {X_train_gru.shape}")

    X_train = np.concatenate([X_train_hand, X_train_gru], axis=1)
    X_dev = np.concatenate([X_dev_hand, X_dev_gru], axis=1)
    print(f"  stacked shape: {X_train.shape}")

    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)

    print("\nTraining stacked XGBClassifier...")
    t0 = time.time()
    clf = XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=50)
    print(f"  {time.time()-t0:.1f}s")

    dev_probs = clf.predict_proba(X_dev)[:, 1]
    ll = log_loss(y_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6))

    with open("model.pkl", "rb") as f:
        old_clf = pickle.load(f)["intent"]
    old_probs = old_clf.predict_proba(X_dev_hand)[:, 1]
    old_ll = log_loss(y_dev, np.clip(old_probs, 1e-6, 1 - 1e-6))

    print(f"\n  Old BCE (hand features only): {old_ll:.4f}")
    print(f"  New BCE (stacked):            {ll:.4f}")
    print(f"  Improvement:                  {old_ll - ll:.4f}")

    if ll < old_ll:
        print("\n  Stacked model is better! Saving...")
        with open("model.pkl", "wb") as f:
            pickle.dump({"intent": clf, "stacked": True, "hand_dim": X_train_hand.shape[1]}, f)
    else:
        print("\n  Stacked model is WORSE. Keeping old model.")


if __name__ == "__main__":
    main()
