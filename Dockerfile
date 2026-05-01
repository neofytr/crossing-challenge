FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install inference dependencies (no torch — GRU inference via onnxruntime)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

COPY predict.py grade.py trajectory_model.py ./
COPY model.pkl model_config.json traj_xgb.pkl ./
COPY best_model_s42.pt best_model_s123.pt best_model_s456.pt ./
COPY model_s42.onnx model_s123.onnx model_s456.onnx ./
COPY model_s42.onnx.data model_s123.onnx.data model_s456.onnx.data ./

ENTRYPOINT ["python", "grade.py"]
