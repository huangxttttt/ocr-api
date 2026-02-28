FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ocr-deepseek --no-install-project

COPY app ./app
COPY main.py ./
COPY .env.example ./

RUN uv sync --frozen --no-dev --extra ocr-deepseek

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
