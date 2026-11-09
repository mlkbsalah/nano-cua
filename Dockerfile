FROM python:3.13-slim-trixie

WORKDIR /workspace

RUN apt-get update && apt-get install -y tmux && apt-get clean

RUN pip install --no-cache-dir uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache
