# syntax=docker/dockerfile:1
#
# Production image for eirel-ai. Builds from this repo's root; the eirel
# SDK is pulled from PyPI via the pyproject.toml dependency list.

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/eirel-ai

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io kubernetes-client \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/eirel-ai

RUN pip install --no-cache-dir /app/eirel-ai

WORKDIR /app/eirel-ai
