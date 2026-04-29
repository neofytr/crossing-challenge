"""Trains the intentionally-naive baseline: XGBoost on engineered tracklet
features for intent, plus constant-velocity trajectory (done at predict time).

Writes model.pkl. Run once:

    python baseline.py

This baseline is deliberately weak — it's the bar to beat, not the finish
line.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from predict import _engineered_features

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"


REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row: pd.Series) -> dict:
    return {k: row[k] for k in REQUEST_FIELDS}


def featurize(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
    return X


def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    print(f"  positive rates: train {train.will_cross_2s.mean():.3f}, "
          f"dev {dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train.shape}")

    pos_ratio = float(y_train.mean())

    print("\nTraining XGBClassifier (no class rebalancing — want calibrated probs)...")
    t0 = time.time()
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
    print(f"  {time.time() - t0:.1f}s")

    dev_probs = clf.predict_proba(X_dev)[:, 1]
    ll = log_loss(y_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6))
    prior_ll = log_loss(y_dev, np.full_like(dev_probs, pos_ratio))
    print(f"\nDev log-loss:  {ll:.4f}  (class-prior baseline {prior_ll:.4f})")

    print(f"\nSaving model → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"intent": clf}, f)


if __name__ == "__main__":
    main()
    sys.exit(0)
