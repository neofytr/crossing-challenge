# Crossing Challenge — Submission

## Final score

Full-dev composite score: **0.6333** (6,065 samples)

- Intent BCE: 0.1921 (baseline: 0.2488)
- Trajectory mean ADE: 24.6 px (baseline: 49.8 px)
- 36.7% improvement over zero-work baseline

---

## Approach

Hybrid architecture combining CatBoost for intent classification with a bidirectional GRU ensemble for trajectory prediction, topped with XGBoost per-horizon blending.

**Trajectory model:** 2-layer bidirectional GRU (hidden_dim=128, input_dim=14, ~546k params) with velocity cumsum parameterization — the model predicts per-horizon velocity increments integrated via `cumsum` to produce smooth displacement trajectories. Trained on 87,684 windows (combined train+eval) with Huber loss (delta=15.0), horizon-weighted objectives (H3: 1.5x, H4: 2.0x), mixup augmentation (30% chance, Beta(0.4,0.4)), and speed perturbation (30% chance, 0.85-1.15x). Each timestep has 14 features: normalized bbox center/size, ego-compensated velocity, ego speed/yaw, compensated acceleration, raw velocity, and ego-induced displacement. Three models (seeds 42, 123, 456) ensembled with test-time horizontal flip augmentation (6 predictions averaged). XGBoost trajectory regressors blend hand-crafted features with GRU predictions at per-horizon optimal weights (H1: 0.73, H2: 0.55, H3: 0.39, H4: 0.29).

**Intent model:** CatBoost classifier with 52 engineered features (47 base + 5 ego-compensated) — positional, velocity, acceleration, ego vehicle, weather/time, motion dynamics, and ego-motion-corrected kinematics. Hyperparameters tuned via Optuna (150 trials CatBoost vs 150 trials XGBoost; CatBoost won). Intent ensemble: 0.80 * CatBoost + 0.20 * GRU intent head.

**Training data:** Re-sliced JAAD+PIE tracklets with stride=2, plus eval set with full labels, producing 87,684 training windows.

---

## Score progression

| Phase | Composite | BCE    | ADE   | What Changed |
|-------|-----------|--------|-------|--------------|
| 0     | 0.8311    | 0.2129 | 40.2  | XGBoost baseline |
| 4     | 0.7123    | 0.2129 | 27.3  | BiGRU trajectory model |
| 8     | 0.6507    | 0.1970 | 25.4  | XGBoost trajectory blending |
| 9     | 0.6388    | 0.1911 | 25.4  | CatBoost intent via Optuna |
| 10    | 0.6387    | 0.1917 | 25.3  | Ego-motion compensation + retune |
| 11    | 0.6350    | 0.1903 | 25.1  | Intent ensemble + deeper XGB blend |
| 15    | 0.6239    | 0.1867 | 24.8  | Combined data + cumsum + mixup + retune |

---

## What didn't work

1. **Polynomial trajectory extrapolation** — quadratic fit on the 16-frame history amplified noise. ADE 54.1 vs 40.2 px.
2. **Larger GRU + temporal attention** — hidden_dim=256 with learned attention overfitted. No improvement over simpler model.
3. **Constant-velocity residual skip** — ADE regressed from 27.3 to 31.6 px. GRU already learns velocity patterns.
4. **GRU encoder stacking for intent** — 256-dim neural features as XGBoost input. BCE worsened from 0.2011 to 0.2087.
5. **XGBoost residual correction** — predicting GRU errors instead of blending. Marginal vs blend approach.
6. **Cross-attention trajectory decoder** — horizon queries with multi-head attention. ADE worse than MLP head.
7. **Gaussian NLL loss** — model exploits variance to minimize loss without improving point estimates. BCE explodes.
8. **5-seed ensemble** — no gain over 3 seeds after XGBoost blending absorbs variance.
9. **XGBoost meta-learner** — GRU predictions as features. ADE 25.7 vs blend 25.0. GRU overfit on train data leaks through.
10. **hidden_dim=192** — overfitting, worse ADE.
11. **Loss rebalancing** [0.5, 1.0, 2.0, 4.0] + Huber delta 30 — ADE 26.9 vs 26.3, worse.
12. **Bbox size features** (dw/dh) — redundant, GRU learns diffs from w/h implicitly.
13. **INTENT_WEIGHT=20** — no change in ADE vs default 50.
14. **Transformer encoder** replacing BiGRU — ADE 27.4 vs 26.3 for 16-step sequences.

---

## Where AI tooling sped me up most

Used **Claude Code** throughout. Biggest acceleration was in the experiment loop — generating training scripts, loss functions, and data pipelines in minutes rather than hours. Also caught the Docker `requirements.txt` bug (CUDA index URL that would have silently blown the image past 2GB). The tool was weakest at architectural intuition — it confidently suggested the polynomial and CV-residual approaches that both failed. The iteration speed still paid off because failures were cheap to test.

---

## How to reproduce

```bash
# Setup
git clone <this-repo>
cd crossing-challenge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Download extended training data (JAAD + PIE annotations)
mkdir -p data/raw
git clone --depth 1 https://github.com/ykotseruba/JAAD.git data/raw/JAAD
git clone --depth 1 https://github.com/aras62/PIE.git data/raw/PIE
cd data/raw/PIE/annotations && unzip annotations.zip && unzip annotations_vehicle.zip && cd ../../../..
python -c "import secrets; print(secrets.token_hex(16))" > .hash_salt
python data/build_tracklets.py
# Edit data/build_windows.py: change STRIDE = 5 to STRIDE = 2
python data/build_windows.py
cp data/dev_original.parquet data/dev.parquet  # restore original dev set

# Combine train + eval for full training set
python -c "
import pandas as pd
train = pd.read_parquet('data/train.parquet')
ev = pd.read_parquet('data/eval.parquet')
pd.concat([train, ev], ignore_index=True).to_parquet('data/train_full.parquet')
"

# Train intent model
python baseline.py

# Train trajectory models (3 seeds)
python train.py --seed 42 --output best_model_s42.pt
python train.py --seed 123 --output best_model_s123.pt
python train.py --seed 456 --output best_model_s456.pt

# Train trajectory XGB blending
python traj_xgb.py

# Tune intent classifier (CatBoost vs XGBoost via Optuna)
pip install catboost optuna
python tune_intent.py

# Score
python grade.py

# Docker
docker build -t my-crossing .
docker run --rm --network none -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```

---

## External data / pretrained weights

- **JAAD** (York University, MIT license): pedestrian action annotations from dashcam video. Used to generate additional training windows. https://github.com/ykotseruba/JAAD
- **PIE** (York University, MIT license): pedestrian intent annotations with OBD ego-motion data. Bulk of training data. https://github.com/aras62/PIE

No pretrained model weights were used. All models trained from scratch on the provided + re-sliced data.
