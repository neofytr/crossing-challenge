from __future__ import annotations

import pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss

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
    return X


def objective_catboost(trial, X_train, y_train, X_dev, y_dev):
    from catboost import CatBoostClassifier

    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced", "None"]),
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    if params["auto_class_weights"] == "None":
        params.pop("auto_class_weights")

    clf = CatBoostClassifier(**params)
    clf.fit(X_train, y_train, eval_set=(X_dev, y_dev), verbose=0)

    probs = clf.predict_proba(X_dev)[:, 1]
    return log_loss(y_dev, np.clip(probs, 1e-6, 1 - 1e-6))


def objective_xgb(trial, X_train, y_train, X_dev, y_dev):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=2000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)

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
        old_data = pickle.load(f)
    old_clf = old_data["intent"]
    old_probs = old_clf.predict_proba(X_dev)[:, 1]
    old_bce = log_loss(y_dev, np.clip(old_probs, 1e-6, 1 - 1e-6))
    print(f"\nCurrent XGBoost BCE: {old_bce:.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n--- Tuning CatBoost (150 trials) ---")
    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(
        lambda trial: objective_catboost(trial, X_train, y_train, X_dev, y_dev),
        n_trials=150,
    )
    print(f"Best CatBoost BCE: {study_cb.best_value:.4f}")
    print(f"Best params: {study_cb.best_params}")

    print("\n--- Tuning XGBoost (150 trials) ---")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, X_dev, y_dev),
        n_trials=150,
    )
    print(f"Best XGBoost BCE: {study_xgb.best_value:.4f}")
    print(f"Best params: {study_xgb.best_params}")

    if study_cb.best_value <= study_xgb.best_value:
        print(f"\n==> CatBoost wins ({study_cb.best_value:.4f} vs {study_xgb.best_value:.4f})")
        from catboost import CatBoostClassifier
        best_params = study_cb.best_params.copy()
        if best_params.get("auto_class_weights") == "None":
            best_params.pop("auto_class_weights")
        best_params.update({
            "iterations": 2000, "loss_function": "Logloss",
            "eval_metric": "Logloss", "random_seed": 42,
            "verbose": 0, "early_stopping_rounds": 50,
        })
        clf = CatBoostClassifier(**best_params)
        clf.fit(X_train, y_train, eval_set=(X_dev, y_dev), verbose=100)
        engine = "catboost"
    else:
        print(f"\n==> XGBoost wins ({study_xgb.best_value:.4f} vs {study_cb.best_value:.4f})")
        from xgboost import XGBClassifier
        bp = study_xgb.best_params.copy()
        clf = XGBClassifier(
            n_estimators=2000, tree_method="hist", n_jobs=-1,
            eval_metric="logloss", early_stopping_rounds=50, **bp,
        )
        clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=100)
        engine = "xgboost"

    final_probs = clf.predict_proba(X_dev)[:, 1]
    final_bce = log_loss(y_dev, np.clip(final_probs, 1e-6, 1 - 1e-6))

    print(f"\n  Old BCE:   {old_bce:.4f}")
    print(f"  New BCE:   {final_bce:.4f} ({engine})")
    print(f"  Delta:     {old_bce - final_bce:.4f}")

    if final_bce < old_bce:
        print(f"\n  Saving new model...")
        with open("model.pkl", "wb") as f:
            pickle.dump({"intent": clf, "stacked": False, "engine": engine}, f)
        print("  Saved!")
    else:
        print("  New model not better. Keeping old.")

    print("\n--- Temperature scaling ---")
    best_temp = 1.0
    best_temp_bce = final_bce
    logits = np.log(np.clip(final_probs, 1e-6, 1-1e-6) / np.clip(1 - final_probs, 1e-6, 1-1e-6))
    for T in np.arange(0.7, 1.31, 0.02):
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        bce = log_loss(y_dev, np.clip(scaled, 1e-6, 1-1e-6))
        if bce < best_temp_bce:
            best_temp_bce = bce
            best_temp = T

    print(f"  Best temperature: {best_temp:.2f}")
    print(f"  BCE with temp scaling: {best_temp_bce:.4f} (was {final_bce:.4f})")

    if best_temp != 1.0 and best_temp_bce < final_bce:
        print(f"  Temperature scaling helps! Saving temp={best_temp:.2f}")
        with open("model.pkl", "rb") as f:
            data = pickle.load(f)
        data["temperature"] = best_temp
        with open("model.pkl", "wb") as f:
            pickle.dump(data, f)

    print("\nDone. Run python grade.py")


if __name__ == "__main__":
    main()
