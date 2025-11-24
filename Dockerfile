# Dockerfile pour déployer l'API ML (dossier ml/)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements du dossier ml et installer
COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY ml/ ./

# Par défaut Railway injecte $PORT
ENV PORT=8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
