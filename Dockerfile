# Dockerfile pour déployer l'API ML (dossier ml/)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements du dossier ml et installer une version CPU compacte de torch
COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu \
    && pip install --no-cache-dir -r requirements.txt --no-deps

# Copier le code de l'API
COPY ml/ ./

# Par défaut Railway injecte $PORT
ENV PORT=8000

CMD ["python", "start.py"]
