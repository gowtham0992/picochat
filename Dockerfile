FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PICOCHAT_RUNS_DIR=/workspace/runs

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY docs ./docs

RUN python -m pip install --no-cache-dir -e ".[hf,monitor]"

RUN useradd --create-home --shell /bin/bash pico
RUN mkdir -p /workspace/runs && chown -R pico:pico /workspace /app
USER pico

EXPOSE 8765 8000

# The web UI binds 0.0.0.0 here, so it requires an auth token. Set one
# explicitly for a stable URL: `-e PICOCHAT_AUTH_TOKEN=...` (otherwise the
# server mints one and prints it to the container logs at startup).
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).status==200 else 1)"

CMD ["pico", "web", "--runs-dir", "/workspace/runs", "--host", "0.0.0.0", "--port", "8765"]
