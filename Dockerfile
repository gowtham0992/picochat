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

CMD ["pico", "web", "--runs-dir", "/workspace/runs", "--host", "0.0.0.0", "--port", "8765"]
