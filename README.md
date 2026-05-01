# Crossing Challenge — Submission

## Final score

Dev composite score: **0.6207** (full dev, 6,065 samples — measured before blind retraining on all data)

- Intent BCE: 0.1909 (baseline floor: 0.2488)
- Trajectory mean ADE: 23.6 px (baseline floor: 49.8 px)
- 37.9% improvement over zero-work baseline

The submitted models are retrained on the full 93,749-sample dataset (train + eval + dev combined) for maximum Eval performance. The 0.6207 score above is the last clean Dev measurement before that blind retrain.

---

## Approach

Hybrid architecture combining CatBoost for intent classification with a bidirectional GRU ensemble for trajectory prediction, topped with intent-conditioned XGBoost per-horizon blending.

**Trajectory model:** 2-layer bidirectional GRU (hidden_dim=128, input_dim=14, ~547k params) with velocity cumsum parameterization — the model predicts per-horizon velocity increments integrated via `cumsum` for smooth displacement trajectories. 14 input features per timestep: normalized bbox center/size, ego-compensated velocity, ego speed/yaw, compensated acceleration, raw velocity, and ego-induced displacement. Three models (seeds 42, 123, 456) ensembled with test-time horizontal flip augmentation (6 forward passes averaged). GRU models pretrained on unlabeled tracklet windows via SSL (masked velocity prediction on ~300k+ windows) then fine-tuned on supervised labels. XGBoost trajectory regressors (67 features: 52 engineered + 8 polynomial/velocity + 6 physics-derived + intent probability) blend with GRU predictions at per-horizon optimized weights.

**Intent model:** CatBoost + LightGBM ensemble with 52 engineered features (47 base + 5 ego-motion-compensated). Features cover position, velocity, acceleration, ego vehicle motion, weather/time flags, motion dynamics, and compensated kinematics. Hyperparameters tuned via Optuna (150 CatBoost + 150 XGBoost trials; CatBoost won). Final intent: 59% CatBoost + 41% LightGBM.

**Class imbalance (7-9% positive):** Handled through calibrated log-loss (CatBoost naturally produces calibrated probabilities), horizon-weighted trajectory loss (H3: 1.5x, H4: 2.0x to emphasize harder long-horizon predictions where crossing pedestrians diverge most), and mixup augmentation (30% chance, Beta(0.4, 0.4)).

**Ego-motion compensation:** Estimated ego-induced bbox displacement from focal length approximation (f_est = fw * 0.7), pedestrian depth from bbox height, and OBD speed/yaw. Raw velocity minus ego displacement gives intrinsic pedestrian motion.

**Training data:** Re-sliced JAAD+PIE tracklets at stride=2, producing 87,684 training windows.

---

## Score progression

| Phase | Composite | BCE    | ADE   | What Changed |
|-------|-----------|--------|-------|--------------|
| 0     | 0.8311    | 0.2129 | 40.2  | XGBoost baseline |
| 4     | 0.7123    | 0.2129 | 27.3  | BiGRU trajectory model |
| 8     | 0.6507    | 0.1970 | 25.4  | XGBoost trajectory blending |
| 9     | 0.6388    | 0.1911 | 25.4  | CatBoost intent via Optuna |
| 10    | 0.6387    | 0.1917 | 25.3  | Ego-motion compensation |
| 11    | 0.6350    | 0.1903 | 25.1  | Intent ensemble + deeper XGB blend |
| 15    | 0.6323    | 0.1921 | 24.5  | Combined data + cumsum + mixup + intent-conditioned XGB |
| 16    | 0.6207    | 0.1909 | 23.6  | LightGBM intent ensemble + SSL GRU pretraining + 6 new XGB features |

---

## What didn't work

1. **Polynomial trajectory extrapolation** — quadratic fit on the 16-frame history amplified noise. ADE 54.1 vs 40.2 px baseline.
2. **Larger GRU + temporal attention** — hidden_dim=256 with learned attention overfitted, no improvement over simpler 128-dim model.
3. **Constant-velocity residual skip** — ADE regressed 27.3 to 31.6 px. GRU already learns velocity implicitly; forcing a CV prior hurts.
4. **Cross-attention trajectory decoder** — horizon queries with multi-head attention. ADE worse than simple MLP head for 16-step sequences.
5. **Gaussian NLL loss** — model exploits variance to minimize NLL without improving point estimates. BCE explodes.
6. **5-seed ensemble** — no gain over 3 seeds after XGBoost blending absorbs inter-seed variance.
7. **XGBoost meta-learner** — GRU predictions as features. ADE 25.7 vs blend 25.0 — GRU overfitting on train data leaks through to the meta-learner.
8. **Transformer encoder** replacing BiGRU — ADE 27.4 vs 26.3. Self-attention is overkill for fixed 16-step sequences.
9. **GRU fine-tuning with 3x crossing weight** — reweighting loss to emphasize crossing pedestrians. No ADE improvement; model already well-optimized.
10. **Camera calibration from PIE intrinsics** (f=1004.8) — GRU had adapted to the approximate f_est=0.7*fw; changing it hurt.
11. **5-seed GRU ensemble after XGBoost blending** — no gain over 3 seeds; XGBoost absorbs inter-seed variance.
12. **XGBoost meta-learner with GRU predictions as features** — ADE 25.7 vs blend 25.0; GRU train-set overfitting leaks through.
13. **hidden_dim=192** — overfitting, worse ADE on dev.

---

## Where AI tooling sped me up most

Used **Claude Code** throughout. Biggest wins: experiment loop velocity (training scripts, loss functions, feature engineering in minutes), catching the Docker CUDA index URL bug that would have bloated the image past 2 GB, and the SSL pretraining data pipeline. The ego-motion compensation derivation and XGBoost blending approach also came out of AI-assisted iteration. Weakest at architectural intuition — polynomial extrapolation and CV-residual skip were confidently suggested and both failed. Iteration speed paid off since failures were cheap to test and discard.

---

## Next experiments

1. **Intent-conditioned trajectory heads** — separate prediction paths for crossing vs non-crossing pedestrians, rather than a single shared trajectory decoder.
2. **Social context features** — nearby pedestrian count, group dynamics, relative positioning (not available in current data but would help with "pedestrian changed their mind" cases).
3. **Multi-modal prediction** — predict multiple plausible futures and select the most likely; current model averages over modes.
4. **Temporal attention with variable-length history** — use the full available history instead of fixed 16 frames, with attention to weight recent frames higher.
5. **Longer SSL pretraining** — more epochs and a stronger masking ratio on the encoder pretraining task.

---

## How to reproduce

```bash
# Setup
git clone https://github.com/neofytr/crossing-challenge.git
cd crossing-challenge
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements-dev.txt

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

# Train intent model (baseline, then Optuna-tuned CatBoost)
python training/baseline.py
pip install catboost optuna
python training/tune_intent.py

# SSL pretrain GRU encoder on unlabeled tracklets
python training/pretrain.py  # outputs training/pretrained_encoder.pt

# Fine-tune GRU trajectory models (3 seeds, requires GPU)
python training/train_pretrained.py  # uses training/pretrained_encoder.pt, outputs best_model_s{42,123,456}.pt

# Train XGBoost trajectory blending (67 features, uses intent probability as feature)
python training/traj_xgb.py

# Score
python grade.py

# Docker build & test
docker build -t my-crossing .
docker run --rm --network none -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```

---

## External data / pretrained weights

- **JAAD** (York University, MIT license): pedestrian action annotations from dashcam video. Used to generate additional training windows via stride-2 re-slicing. https://github.com/ykotseruba/JAAD
- **PIE** (York University, MIT license): pedestrian intent annotations with OBD ego-motion data. Bulk of training data. https://github.com/aras62/PIE

No pretrained model weights were used. All models trained from scratch on the provided + re-sliced data.

---

_Total time spent on this challenge: ~30 hours._
