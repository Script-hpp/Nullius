# Nullius: API (FastAPI) und gebaute Web-App in einem Container. Keine Schluessel im Image: alles ueber Umgebungsvariablen (deploy/.env.example).
#   docker build -t nullius .
#   docker run --rm -p 8000:8000 --env-file deploy/.env -v nullius-data:/data -v nullius-logs:/app/logs nullius
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" \
    REPRO_EXPORT_DIR=/data WEB_DIST=/app/web/dist
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev && .venv/bin/python -c "from repro_agent.loop import Orchestrator"
COPY benchmark/ benchmark/
COPY deploy/serve.py deploy/serve.py
COPY --from=web /web/dist web/dist
RUN mkdir -p /data logs runs && useradd --create-home --uid 10001 app && chown -R app /app /data
USER app
VOLUME ["/data", "/app/logs", "/app/runs"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--no-server-header", "--limit-concurrency", "200", "--timeout-keep-alive", "15"]
