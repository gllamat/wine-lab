# Reproducible environment for the wine/WSET pipeline. Python 3.11, pinned requirements.
#   docker build --target core  -t winelab:core  .      # matcher, terroir, dashboard build, blind-tasting scoring
#   docker build --target embed -t winelab:embed .      # + MiniLM/BERT embeddings (CPU torch)
#   docker build --target causal -t winelab:causal .    # + statsmodels for the causal module
#   docker build --target serve -t winelab:serve .      # + HTTP API for the "describe a wine" endpoint
# Docker Hub can be slow/blocked on some networks: the AWS public mirror serves the same official image.
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.11-slim
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NUMBA_CACHE_DIR=/tmp/numba
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS core
COPY requirements/core.txt requirements/core.txt
RUN pip install -r requirements/core.txt
COPY lexicon lexicon
COPY src src
COPY scripts scripts
COPY dashboard dashboard
# data lives OUTSIDE the image: mount ../datasets, outputs, cache (see docker-compose.yml)
ENV WINE_DATASETS=/data/datasets
CMD ["python", "scripts/smoke.py"]

FROM core AS causal
COPY requirements/causal.txt requirements/causal.txt
RUN pip install -r requirements/causal.txt

FROM core AS embed
COPY requirements/embed.txt requirements/embed.txt
RUN pip install -r requirements/embed.txt --extra-index-url https://download.pytorch.org/whl/cpu
ENV HF_HOME=/app/cache/hf

FROM embed AS serve
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
