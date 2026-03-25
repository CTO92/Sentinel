# SENTINEL Training Monitor — Docker Build
#
# Lightweight Python image with non-root user.
#
# Build:
#   docker build -t sentinel-training-monitor -f training-monitor/packaging/Dockerfile .

FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir build

COPY training-monitor/pyproject.toml /build/
COPY training-monitor/src/ /build/src/

RUN python -m build --wheel --outdir /dist

# ============================================================================
#  Runtime stage
# ============================================================================
FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r sentinel && useradd -r -g sentinel -m -d /home/sentinel sentinel

# Install the wheel
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Copy configuration defaults
COPY config/ /etc/sentinel/ 2>/dev/null || true

USER sentinel
WORKDIR /home/sentinel

EXPOSE 9090

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:9090/metrics || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "sentinel_training"]
