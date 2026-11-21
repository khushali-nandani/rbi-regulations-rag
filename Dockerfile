# syntax=docker/dockerfile:1
# Stage 1 builds the corpus (ingest -> Delta -> Qdrant index) and downloads the embedding models,
# so the running container needs no network access for retrieval and starts in seconds.
FROM python:3.12-slim AS builder
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/build-venv PYTHONPATH=/app/src
RUN pip install --no-cache-dir uv==0.10.10
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra pipeline --no-install-project
COPY src ./src
COPY data/manifest.yaml ./data/manifest.yaml
COPY data/raw ./data/raw
RUN /opt/build-venv/bin/python -m rbirag.ingest.pipeline \
 && /opt/build-venv/bin/python -m rbirag.retrieval.index --serving-only

# Stage 2: runtime only (no pandas / deltalake / pymupdf / mlflow), non-root user.
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONPATH=/app/src PATH=/opt/venv/bin:$PATH \
    PORT=7860 HF_HUB_OFFLINE=1
RUN pip install --no-cache-dir uv==0.10.10
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && pip uninstall -y uv
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app src ./src
COPY --from=builder --chown=app:app /app/qdrant_local ./qdrant_local
COPY --from=builder --chown=app:app /app/.cache/fastembed ./.cache/fastembed
RUN mkdir -p /app/traces /app/.cache/llm && chown app:app /app/traces /app/.cache /app/.cache/llm
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'), timeout=4)"
CMD ["sh", "-c", "exec uvicorn rbirag.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
