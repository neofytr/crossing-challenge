FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py grade.py trajectory_model.py ./
COPY model.pkl model_config.json ./
COPY best_model_s42.pt best_model_s123.pt best_model_s456.pt ./

ENTRYPOINT ["python", "grade.py"]
