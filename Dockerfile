FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r requirements.txt

COPY src ./src

RUN useradd -m -u 1000 appuser \
 && mkdir -p /app/.cache/huggingface \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/ || exit 1

CMD ["python3", "-m", "src.app"]
