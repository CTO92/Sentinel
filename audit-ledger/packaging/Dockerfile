# SENTINEL Audit Ledger — Multi-stage Docker build
# Stage 1: Build the Rust binary.

FROM rust:latest AS builder

WORKDIR /build

# Cache dependency compilation by copying manifests first.
COPY audit-ledger/Cargo.toml audit-ledger/Cargo.lock* audit-ledger/build.rs ./
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs && \
    echo 'pub mod compliance; pub mod grpc; pub mod ledger; pub mod storage; pub mod util;' > src/lib.rs && \
    mkdir -p src/compliance src/grpc src/ledger src/storage/migrations src/util && \
    touch src/compliance/mod.rs src/grpc/mod.rs src/ledger/mod.rs src/storage/mod.rs src/util/mod.rs && \
    cargo build --release 2>/dev/null || true

# Copy the real source and rebuild.
COPY audit-ledger/src/ src/
RUN cargo build --release --bin sentinel-audit-ledger

# Stage 2: Minimal runtime image.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --create-home sentinel

COPY --from=builder /build/target/release/sentinel-audit-ledger /usr/local/bin/

USER sentinel
WORKDIR /home/sentinel

# gRPC port
EXPOSE 50052

# Prometheus metrics (served via the gRPC reflection or a sidecar).
EXPOSE 9090

ENV RUST_LOG=info,sqlx=warn

ENTRYPOINT ["sentinel-audit-ledger"]
