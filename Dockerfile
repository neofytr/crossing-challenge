FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install inference dependencies (no torch — GRU inference via onnxruntime)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ORT_NUM_THREADS=4

COPY predict.py grade.py trajectory_model.py ./
COPY models/ ./models/

ENTRYPOINT ["python", "grade.py"]
