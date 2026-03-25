# ==============================================================================
# SENTINEL Probe Agent — Multi-stage Docker build
# ==============================================================================
# Build:   docker build -t sentinel-probe-agent -f probe-agent/packaging/Dockerfile .
# Run:     docker run --gpus all --rm sentinel-probe-agent:latest
# ==============================================================================

# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release
ARG CUDA_ARCHITECTURES="70;75;80;86;89;90"

# Install build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    pkg-config \
    python3 \
    python3-pip \
    libssl-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libspdlog-dev \
    nlohmann-json3-dev \
    libgtest-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install prometheus-cpp from source (not packaged in Ubuntu 22.04).
RUN git clone --depth 1 --branch v1.2.4 \
        https://github.com/jupp0r/prometheus-cpp.git /tmp/prometheus-cpp && \
    cd /tmp/prometheus-cpp && \
    git submodule update --init --recursive && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DENABLE_TESTING=OFF \
        -DENABLE_PUSH=OFF \
        -DENABLE_COMPRESSION=OFF && \
    cmake --build build --target install && \
    rm -rf /tmp/prometheus-cpp

WORKDIR /build

# Copy proto definitions first (for layer caching).
COPY proto/ /build/proto/

# Copy probe agent source.
COPY probe-agent/ /build/probe-agent/

# Build the probe agent.
WORKDIR /build/probe-agent
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_INSTALL_PREFIX=/opt/sentinel \
    -DSENTINEL_BUILD_TESTS=OFF && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libgrpc++1 \
    libprotobuf23 \
    libspdlog1 \
    curl \
    tini \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the agent.
RUN groupadd -r sentinel && useradd -r -g sentinel -d /opt/sentinel sentinel

# Copy built binaries and libraries.
COPY --from=builder /opt/sentinel/ /opt/sentinel/

# Copy prometheus-cpp shared libraries.
COPY --from=builder /usr/local/lib/libprometheus-cpp*.so* /usr/local/lib/
RUN ldconfig

# Create directories for logs and config.
RUN mkdir -p /var/log/sentinel /etc/sentinel && \
    chown -R sentinel:sentinel /var/log/sentinel /etc/sentinel /opt/sentinel

ENV PATH="/opt/sentinel/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/sentinel/lib:${LD_LIBRARY_PATH}"

# Health check: verify the metrics endpoint is responsive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:9101/metrics || exit 1

# Run as non-root.
USER sentinel
WORKDIR /opt/sentinel

# Expose Prometheus metrics port.
EXPOSE 9101

# Use tini as PID 1 for proper signal handling.
ENTRYPOINT ["tini", "--"]
CMD ["sentinel-probe-agent", "-c", "/etc/sentinel/config.json"]

# Labels.
LABEL maintainer="SENTINEL Team"
LABEL description="SENTINEL Probe Agent — GPU SDC detection daemon"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/sentinel-project/sentinel"
