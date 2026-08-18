FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BSAVCI_DATABASE_PATH=/app/data/paper_trading.db \
    BSAVCI_HEARTBEAT_PATH=/app/data/worker-heartbeat.json \
    BSAVCI_HEALTH_PORT=8080

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/data \
    && chown -R app:app /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir \
    -r requirements.txt

COPY --chown=app:app src ./src

USER app

VOLUME ["/app/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s \
    --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=3)"

CMD ["python", "-m", "src.deployment.health"]
