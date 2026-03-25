FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir hatchling

# Copy source
COPY pyproject.toml setup.cfg ./
COPY src/ src/

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# --- Runtime stage ---
FROM python:3.12-slim

LABEL maintainer="SENTINEL Team"
LABEL description="SENTINEL Inference Monitor sidecar for Silent Data Corruption detection"

WORKDIR /app

# Install runtime dependencies only
COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Non-root user
RUN groupadd -r sentinel && useradd -r -g sentinel sentinel
USER sentinel

# Health check
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

# Prometheus metrics port
EXPOSE 9090
# Health check port
EXPOSE 8080

ENTRYPOINT ["sentinel-inference-monitor"]
CMD ["--config", "/etc/sentinel/config.yaml"]
