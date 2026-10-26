FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Weights are baked into the image: run `bash scripts/download_models.sh` and train beforehand.
COPY models/ /app/models/
COPY src/ /app/src/
COPY run.sh /app/run.sh

ENV ASR_MODEL=/app/models/asr LLM_MODEL=/app/models/lifemate-llm
ENTRYPOINT ["bash", "/app/run.sh"]
