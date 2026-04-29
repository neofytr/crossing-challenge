# Crossing Challenge — Submission

## Final score

Dev composite score: **0.6603** (from `python grade.py`)

- Intent BCE: 0.1973 (baseline: 0.2129)
- Trajectory mean ADE: 26.3 px (baseline: 40.2 px)
- 20.5% improvement over baseline

---

## Approach

Hybrid architecture combining XGBoost for intent classification with a bidirectional GRU for trajectory prediction.

**Trajectory model:** 2-layer bidirectional GRU (hidden_dim=128, ~546k params) trained on normalized 16-frame sequences. Each timestep has 8 features: normalized bbox center/size, frame-to-frame velocity, ego speed, and ego yaw. The model predicts center displacements at 4 horizons — only centers affect ADE scoring, so bbox width/height are held constant. Three models trained with different seeds are ensembled with test-time horizontal flip augmentation (6 predictions per sample, averaged).

**Intent model:** XGBoost classifier with 30 engineered features — the original 20 positional/ego features plus 10 motion dynamics features (total displacement, velocity magnitude, heading angle, acceleration, bbox size change rate, area ratio, vertical position). Tuned with early stopping, subsample=0.8, colsample_bytree=0.8.

**Training data:** Re-sliced JAAD+PIE tracklets with stride=2 (vs original stride=5), producing 70,737 training windows — 2.5x more than the starter's 28,680. This helped intent (BCE -3.3%) more than trajectory.

---

## What didn't work

1. **Polynomial trajectory extrapolation** — quadratic fit on the 16-frame history amplified noise at extrapolation distances. Degree-2 polyfit worsened ADE from 40.2 to 54.1 px. The baseline's "mean of last 4 velocity diffs" is a better recency-weighted estimator than any polynomial over the full history.

2. **Larger GRU + temporal attention** — hidden_dim=256 with a learned temporal attention mechanism overfitted on both 28k and 70k samples. The best it achieved was 27.3 px ADE (vs 27.3 with the simpler model). The 16-step input sequence is too short for attention to provide meaningful benefit over the GRU's built-in recency bias.

3. **Constant-velocity residual skip connection** — adding an explicit CV baseline as a skip connection (model predicts correction to CV) regressed ADE from 27.3 to 31.6 px. The GRU already learns velocity patterns from the raw input; the skip connection constrained rather than helped.

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

# Train trajectory models (3 seeds)
python train.py --seed 42 --output best_model_s42.pt
python train.py --seed 123 --output best_model_s123.pt
python train.py --seed 456 --output best_model_s456.pt

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
