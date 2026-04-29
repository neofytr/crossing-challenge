# Crossing Challenge — Submission

## Final score

Dev composite score: **0.6507** (from `python grade.py`)

- Intent BCE: 0.1970 (baseline: 0.2129)
- Trajectory mean ADE: 25.4 px (baseline: 40.2 px)
- 21.7% improvement over baseline

---

## Approach

Hybrid architecture combining XGBoost for intent classification with a bidirectional GRU for trajectory prediction.

**Trajectory model:** 2-layer bidirectional GRU (hidden_dim=128, ~546k params) trained on normalized 16-frame sequences with Huber loss (delta=15.0) and horizon-weighted objectives (H3: 1.5x, H4: 2.0x). Each timestep has 10 features: normalized bbox center/size, frame-to-frame velocity, ego speed, ego yaw, and acceleration (ax, ay). Five models trained with different seeds ensembled with test-time horizontal flip augmentation (10 predictions averaged). XGBoost trajectory regressors trained per-horizon and blended with GRU predictions at optimal weights.

**Intent model:** XGBoost classifier with 47 engineered features — positional, velocity, acceleration, ego vehicle, weather/time, plus motion dynamics (displacement, heading, aspect ratio changes, lateral/longitudinal ratios, stationarity). Tuned with early stopping.

**Training data:** Re-sliced JAAD+PIE tracklets with stride=2 (vs original stride=5), producing 70,737 training windows — 2.5x more than the starter's 28,680. Speed perturbation augmentation (30% chance, scale 0.85-1.15) during GRU training.

---

## What didn't work

1. **Polynomial trajectory extrapolation** — quadratic fit on the 16-frame history amplified noise at extrapolation distances. Degree-2 polyfit worsened ADE from 40.2 to 54.1 px. The baseline's "mean of last 4 velocity diffs" is a better recency-weighted estimator than any polynomial over the full history.

2. **Larger GRU + temporal attention** — hidden_dim=256 with a learned temporal attention mechanism overfitted on both 28k and 70k samples. The best it achieved was 27.3 px ADE (vs 27.3 with the simpler model). The 16-step input sequence is too short for attention to provide meaningful benefit over the GRU's built-in recency bias.

3. **Constant-velocity residual skip connection** — adding an explicit CV baseline as a skip connection (model predicts correction to CV) regressed ADE from 27.3 to 31.6 px. The GRU already learns velocity patterns from the raw input; the skip connection constrained rather than helped.

4. **GRU encoder stacking for intent** — extracted 256-dim GRU encoder features and fed them alongside 47 hand-crafted features to XGBoost. BCE worsened from 0.2011 to 0.2087. High-dimensional neural representations add noise to gradient boosting.

5. **XGBoost residual trajectory correction** — training XGBoost to predict GRU trajectory errors (residuals) instead of raw targets. Residual ADE 25.2 vs blend ADE 25.1. The convex blend approach is slightly better because it constrains the XGBoost to not over-correct.

---

## Where AI tooling sped me up most

Used **Claude Code** throughout. Biggest acceleration was in the experiment loop — generating training scripts, loss functions, and data pipelines in minutes rather than hours. Also caught the Docker `requirements.txt` bug (CUDA index URL that would have silently blown the image past 2GB). The tool was weakest at architectural intuition — it confidently suggested the polynomial and CV-residual approaches that both failed. The iteration speed still paid off because failures were cheap to test.

---

## Next experiments

- **Transformer encoder** replacing GRU — global self-attention over all 16 timesteps with learned positional encoding. Also enables migration to a custom C deep learning framework (Axiom) that has MHA but no RNN primitives.
- **Autoregressive trajectory decoder** — predict each horizon conditioned on the previous prediction, rather than all 4 from a single context vector. Should help long-horizon ADE specifically.
- **GRU encoder stacking for intent** — extract the GRU's learned 256-dim context vector and feed it as additional features to XGBoost. Gives the classifier access to temporal representations that hand-crafted features can't capture.
- **Ego-motion compensated coordinates** — subtract estimated ego-induced pixel displacement from observed trajectories before feeding to the model.

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

# Train intent model
python baseline.py

# Train trajectory models (5 seeds)
python train.py --seed 42 --output best_model_s42.pt
python train.py --seed 123 --output best_model_s123.pt
python train.py --seed 456 --output best_model_s456.pt
python train.py --seed 789 --output best_model_s789.pt
python train.py --seed 1 --output best_model_s1.pt

# Train trajectory XGB blending
python traj_xgb.py

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

---

Total time spent on this challenge: ~8 hours.
