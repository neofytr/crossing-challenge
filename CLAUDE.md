# Crossing Challenge — Gobblecube AI Builder Take-Home

## Task
Predict pedestrian crossing intent (binary) and 2-second future trajectory (4 bbox horizons at 0.5s intervals) from 16-frame dashcam bbox history + ego vehicle telemetry.

## Scoring
```
composite = 0.5 * (BCE / 0.2488) + 0.5 * (mean_ADE / 49.80)
```
Lower is better. 1.0 = zero-work baseline (class prior + zero velocity).

## Current Best Scores (full dev, 6065 samples)
- Composite: 0.6323
- Intent BCE: 0.1921
- Trajectory ADE: 24.5 px

## Score Progression
| Phase | Composite | BCE    | ADE   | What Changed |
|-------|-----------|--------|-------|--------------|
| 0     | 0.8311    | 0.2129 | 40.2  | XGBoost baseline |
| 4     | 0.7123    | 0.2129 | 27.3  | BiGRU trajectory model |
| 8     | 0.6507    | 0.1970 | 25.4  | XGBoost trajectory blending |
| 9     | 0.6388    | 0.1911 | 25.4  | CatBoost intent via Optuna |
| 10    | 0.6387    | 0.1917 | 25.3  | Ego-motion compensation + retune |
| 11    | 0.6350    | 0.1903 | 25.1  | Intent ensemble + deeper XGB blend |
| 15    | 0.6323    | 0.1921 | 24.5  | Combined data + cumsum + mixup + intent-conditioned XGB |

## Data
- **Train**: 70,737 windows (stride=2 re-slicing of JAAD+PIE), **Dev**: 6,065, **Eval**: 16,947 (has full labels)
- **Train+Eval combined**: 87,684 windows (use data/train_full.parquet)
- **Positive rate**: train=7.9%, dev=9.1%, eval=6.8% (heavily imbalanced)
- **Two distinct sources**:
  - PIE (93.6%): ego_available=True, 6.4% positive, no metadata
  - JAAD (6.4%): ego_available=False, 48.2% positive, has time_of_day/weather/location
- **Frame size**: Almost all 1920x1080 (275 samples are 1280x720)
- **Ego yaw outliers**: min=-53.5, max=34.5 rad/s. Must clamp to [-0.5, 0.5] before use.

## Architecture

### Intent: CatBoost classifier
- 52 engineered features (47 base + 5 ego-compensated)
- Tuned via Optuna (150 trials CatBoost vs XGBoost; CatBoost won)
- Stored in `model.pkl`

### Trajectory: BiGRU ensemble + XGBoost blending
- 2-layer BiGRU (hidden=128, input_dim=14, ~546k params)
- 14 features per timestep: [cx, cy, w, h, dx_comp, dy_comp, ego_speed, ego_yaw, ax_comp, ay_comp, dx, dy, ego_dx, ego_dy] (all normalized)
- 3-seed ensemble (42, 123, 456) with TTA horizontal flip = 6 forward passes averaged
- XGBoost per-horizon blending on top (55 features → 8 regressors)
- Stored in `best_model_s{42,123,456}.pt` + `traj_xgb.pkl`

### TTA Flip (14-dim input)
Indices to transform on horizontal flip:
- [0] cx/fw → 1.0 - value
- [4] dx_comp/fw → negate
- [7] ego_yaw → negate
- [8] ax_comp/fw → negate
- [10] dx/fw → negate
- [12] ego_dx/fw → negate

## Key Files
| File | Purpose |
|------|---------|
| `predict.py` | **Submission entry point** — `predict(request) → dict` |
| `grade.py` | Local scoring harness, mirrors Gobblecube grader |
| `trajectory_model.py` | CrossingModel (BiGRU + MLP heads) |
| `trajectory_data.py` | TrajectoryDataset with ego compensation, 14 features |
| `train.py` | GRU training: Huber loss, horizon weights, cosine LR |
| `traj_xgb.py` | XGBoost trajectory blending trainer |
| `tune_intent.py` | Optuna CatBoost/XGBoost intent tuning |
| `baseline.py` | Simple XGBoost intent trainer |
| `model_config.json` | GRU config: `{"input_dim": 14, "hidden_dim": 128, ...}` |
| `Dockerfile` | CPU-only inference container |
| `data/schema.md` | Full data schema documentation |

## Training Hyperparameters
- **GRU**: AdamW lr=8e-4 (cosine decay from 5-epoch warmup), batch=512, epochs=80, patience=15
- **Loss**: Huber(delta=15) in pixel space, horizon weights [1.0, 1.0, 1.5, 2.0] + BCE * 50.0
- **Augmentation**: Horizontal flip 50%, speed perturbation 30% (0.85-1.15x), mixup 30% (Beta 0.4,0.4)
- **Parameterization**: Velocity cumsum (predict increments, integrate via cumsum)

## Baselines (dev set)
- Zero-velocity ADE: 62.5 px
- Constant-velocity ADE: 39.5 px (mean of last 4 velocities × steps_ahead)
- Current best ADE: 24.5 px

## Target Statistics (training set)
- H1 (500ms): mean displacement 22.7 px
- H2 (1000ms): 44.9 px
- H3 (1500ms): 75.8 px
- H4 (2000ms): 110.9 px
- Lateral motion (dx) variance is 10x vertical (dy) variance
- Crossing pedestrians: 189.9 px mean displacement at 2s vs 104.2 px non-crossing

## What Didn't Work
1. Polynomial trajectory extrapolation (ADE worse)
2. Larger GRU + temporal attention (no improvement, overfitting)
3. CV-residual skip connection (ADE 31.6 vs 27.3)
4. GRU encoder stacking for intent (BCE worse)
5. XGBoost residual correction (marginal vs blend)
6. Cross-attention trajectory decoder with horizon queries (ADE worse)
7. Gaussian NLL loss (exploited by model, BCE exploded)
8. 5-seed ensemble (no gain over 3 after XGBoost blending)
9. XGBoost meta-learner with GRU predictions as features (ADE 25.7 vs blend 25.0 — GRU overfit on train leaks through)
10. hidden_dim=192 (overfitting, worse ADE)
11. Loss rebalancing [0.5, 1.0, 2.0, 4.0] + Huber delta 30 (ADE 26.9 vs 26.3 — worse)
12. Bbox size change features dw/dh (redundant — GRU learns diffs from w/h implicitly)
13. INTENT_WEIGHT=20 (no change in ADE)
14. Transformer encoder replacing BiGRU (ADE 27.4 vs 26.3 — worse for 16-step sequences)
15. Camera calibration fix (f_est 1344→1004.8 from PIE intrinsics) — GRU had already adapted to approximate focal length
16. Finer XGBoost blend weight search (0.01 step) — marginal refinement
17. GRU fine-tuning with 3x crossing weight — no ADE improvement, model already well-optimized

## Hints from Problem Statement — Status
- "any pretrained model" — not used; all models trained from scratch
- "additional public datasets" — JAAD+PIE used for stride-2 re-slicing (87k windows)
- PedestrianActionBenchmark (WACV '21) — techniques informed feature engineering
- ENCORE/PedFormer outputs — can't use (hashed IDs, different splits/temporal resolution)
- PIE camera calibration — tried fx=1004.8; GRU adapted to f_est=0.7*fw, no improvement
- Intent-conditioned trajectory — DONE: intent_prob as 61st XGB feature
- train_all.parquet — available for blind final training (93,749 = train+eval+dev)

## Ego-Motion Compensation
Ego-induced bbox displacement estimated per frame:
```python
ego_yaw_c = np.clip(ego_yaw, -0.5, 0.5)
ego_speed_c = np.clip(ego_speed, 0.0, 15.0)
f_est = fw * 0.7  # approximate focal length
depth_t = f_est * 1.7 / max(h[t], 10.0)  # depth from bbox height
ego_dx[t] = f_est * ego_yaw_c[t] * dt + cx_rel * ego_speed_c[t] * dt / depth_t
ego_dy[t] = cy_rel * ego_speed_c[t] * dt / depth_t
```
Compensated velocity: dx_comp = dx - ego_dx

## Commit Style
Short human-like messages, NO Co-Authored-By or AI attribution.

## Docker
```bash
docker build -t my-crossing .
docker run --rm --network none -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```
Container: python:3.11-slim, CPU-only PyTorch, copies model files + predict.py + grade.py + trajectory_model.py.
