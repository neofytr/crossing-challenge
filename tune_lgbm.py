from __future__ import annotations

import pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss
import lightgbm as lgb

from predict import _engineered_features

DATA = Path(__file__).parent / "data"

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
            print(f"  featurize {i}/{n}")
    return X


def objective(trial, X_train, y_train, X_dev, y_dev):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    probs = clf.predict_proba(X_dev)[:, 1]
    return log_loss(y_dev, np.clip(probs, 1e-6, 1 - 1e-6))


def main():
    print("Loading data...")
    train_path = DATA / "train_full.parquet"
    if not train_path.exists():
        train_path = DATA / "train.parquet"
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(DATA / "dev.parquet")

    print("Featurizing...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time()-t0:.1f}s  shape: {X_train.shape}")

    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    cat_clf = model_data["intent"]
    cat_probs = cat_clf.predict_proba(X_dev)[:, 1]
    cat_bce = log_loss(y_dev, np.clip(cat_probs, 1e-6, 1 - 1e-6))
    print(f"\nCurrent CatBoost BCE: {cat_bce:.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n--- Tuning LightGBM (150 trials) ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_dev, y_dev),
        n_trials=150,
    )
    print(f"Best LightGBM BCE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    print("\nRetraining best LightGBM...")
    bp = study.best_params.copy()
    tuned_clf = lgb.LGBMClassifier(
        n_estimators=2000, objective="binary", metric="binary_logloss",
        random_state=42, verbose=-1, n_jobs=-1, **bp,
    )
    tuned_clf.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = tuned_clf.best_iteration_

    final_clf = lgb.LGBMClassifier(
        n_estimators=best_iter, objective="binary", metric="binary_logloss",
        random_state=42, verbose=-1, n_jobs=-1, **bp,
    )
    final_clf.fit(X_train, y_train)

    lgbm_probs = final_clf.predict_proba(X_dev)[:, 1]
    lgbm_bce = log_loss(y_dev, np.clip(lgbm_probs, 1e-6, 1 - 1e-6))
    print(f"LightGBM-only BCE: {lgbm_bce:.4f} (best_iteration={best_iter})")

    print("\nSearching optimal CatBoost/LightGBM blend weight...")
    best_bce = float("inf")
    best_w_cat = 1.0
    for w_cat in np.arange(0.0, 1.01, 0.01):
        blended = w_cat * cat_probs + (1 - w_cat) * lgbm_probs
        bce = log_loss(y_dev, np.clip(blended, 1e-6, 1 - 1e-6))
        if bce < best_bce:
            best_bce = bce
            best_w_cat = w_cat

    w_lgbm = round(1 - best_w_cat, 2)
    print(f"  Best w_cat={best_w_cat:.2f}, w_lgbm={w_lgbm:.2f}")

    print(f"\n{'='*50}")
    print(f"  CatBoost-only BCE:  {cat_bce:.4f}")
    print(f"  LightGBM-only BCE:  {lgbm_bce:.4f}")
    print(f"  Blended BCE:        {best_bce:.4f}")
    print(f"  Delta (cat - blend): {cat_bce - best_bce:.4f}")

    if best_bce < cat_bce:
        print(f"\nBlend wins! Saving model.pkl with lgbm (w_lgbm={w_lgbm:.2f})...")
        model_data["lgbm"] = final_clf
        model_data["lgbm_weight"] = w_lgbm
        model_data["engine"] = "catboost+lgbm"
        with open("model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        print("Saved!")
    else:
        print("\nBlend does NOT improve over CatBoost-only. Not saving.")

    print("\nDone.")


if __name__ == "__main__":
    main()
