FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (before requirements.txt)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install inference dependencies only (no CUDA, no dev tools)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py grade.py trajectory_model.py ./
COPY model.pkl model_config.json traj_xgb.pkl ./
COPY best_model_s42.pt best_model_s123.pt best_model_s456.pt ./

ENTRYPOINT ["python", "grade.py"]
