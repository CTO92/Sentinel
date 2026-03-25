# SENTINEL Correlation Engine — Multi-stage Docker Build
#
# Build stage: compile Rust binary with static linking
# Runtime stage: minimal Debian image
#
# Build:
#   docker build -t sentinel-correlation-engine -f correlation-engine/packaging/Dockerfile .

# ============================================================================
#  Build stage
# ============================================================================
FROM rust:latest AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        libprotobuf-dev \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for dependency caching
COPY correlation-engine/Cargo.toml correlation-engine/Cargo.lock* /build/correlation-engine/
COPY correlation-engine/build.rs /build/correlation-engine/

# Copy proto definitions
COPY proto/ /build/proto/

# Create dummy main for dependency caching
RUN mkdir -p /build/correlation-engine/src && \
    echo "fn main() {}" > /build/correlation-engine/src/main.rs && \
    mkdir -p /build/correlation-engine/benches && \
    echo "fn main() {}" > /build/correlation-engine/benches/dummy.rs

WORKDIR /build/correlation-engine
RUN cargo build --release 2>/dev/null || true

# Copy actual source and rebuild
COPY correlation-engine/src/ /build/correlation-engine/src/
COPY correlation-engine/benches/ /build/correlation-engine/benches/

# Touch main.rs to trigger rebuild of the actual source
RUN touch /build/correlation-engine/src/main.rs 2>/dev/null || \
    touch /build/correlation-engine/src/lib.rs 2>/dev/null || true

RUN cargo build --release

# ============================================================================
#  Runtime stage
# ============================================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r sentinel && useradd -r -g sentinel -m -d /opt/sentinel sentinel

# Copy binary
COPY --from=builder /build/correlation-engine/target/release/correlation-engine /usr/local/bin/sentinel-correlation-engine
RUN chmod +x /usr/local/bin/sentinel-correlation-engine

# Configuration is mounted at runtime via volumes or ConfigMaps.
# Create the config directory for the mount point.
RUN mkdir -p /etc/sentinel

RUN mkdir -p /var/lib/sentinel && chown sentinel:sentinel /var/lib/sentinel

USER sentinel
WORKDIR /opt/sentinel

EXPOSE 50051 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["sentinel-correlation-engine"]
