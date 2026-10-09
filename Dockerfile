# Multi-stage build: dependencies are resolved with uv from the lockfile in a
# build stage; the runtime stage only gets the virtual environment.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
# libgomp: OpenMP runtime needed by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 halftick
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY config.yaml ./
ENV PATH=/app/.venv/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1
USER halftick
EXPOSE 8000
ENTRYPOINT ["halftick"]
CMD ["replay", "--realtime", "--metrics-port", "8000", "--start", "09:30", "--loop"]
