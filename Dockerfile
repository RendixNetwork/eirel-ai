# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/eirel-ai

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io kubernetes-client \
    && rm -rf /var/lib/apt/lists/*

COPY eirel-ai /app/eirel-ai

RUN pip install --no-cache-dir "eirel>=0.2.0" \
    && pip install --no-cache-dir /app/eirel-ai

WORKDIR /app/eirel-ai

FROM python:3.12-slim AS test
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/eirel-ai

COPY eirel-ai /app/eirel-ai

RUN pip install --no-cache-dir "eirel>=0.2.0" \
    && pip install --no-cache-dir /app/eirel-ai \
    && pip install pytest pytest-asyncio

WORKDIR /app/eirel-ai
