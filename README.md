# SENTINEL

**Silent Data Corruption Detection Framework for GPU Clusters**

[![CI - Probe Agent](https://github.com/sentinel-sdc/sentinel/actions/workflows/ci-probe-agent.yml/badge.svg)](https://github.com/sentinel-sdc/sentinel/actions/workflows/ci-probe-agent.yml)
[![CI - Correlation Engine](https://github.com/sentinel-sdc/sentinel/actions/workflows/ci-correlation-engine.yml/badge.svg)](https://github.com/sentinel-sdc/sentinel/actions/workflows/ci-correlation-engine.yml)
[![CI - Integration](https://github.com/sentinel-sdc/sentinel/actions/workflows/ci-integration.yml/badge.svg)](https://github.com/sentinel-sdc/sentinel/actions/workflows/ci-integration.yml)
[![Security Scan](https://github.com/sentinel-sdc/sentinel/actions/workflows/security-scan.yml/badge.svg)](https://github.com/sentinel-sdc/sentinel/actions/workflows/security-scan.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status: Pre-release Alpha](https://img.shields.io/badge/Status-Pre--release%20Alpha-orange.svg)](#status)

> **Pre-release Alpha Notice:** SENTINEL is currently in pre-release alpha (v0.1.0-alpha). APIs, configuration formats, and wire protocols are subject to breaking changes. This software is made available for early evaluation, testing, and community feedback. It is **not yet recommended for production use**. Please report issues and share feedback via [GitHub Issues](https://github.com/sentinel-sdc/sentinel/issues).

SENTINEL is an open-source framework for detecting Silent Data Corruption (SDC) in large-scale GPU clusters, licensed under the [Apache License 2.0](LICENSE). SDC occurs when hardware faults produce incorrect computation results without raising errors, leading to silently corrupted model outputs. SENTINEL provides multi-layered detection across probe agents, inference/training monitors, and a fleet-wide correlation engine with full audit trails.

## Architecture

```
+------------------------------------------------------------------+
|                        GPU Cluster                                |
|                                                                   |
|  +-------------+  +-------------+  +-------------+               |
|  |   GPU Node  |  |   GPU Node  |  |   GPU Node  |  ...          |
|  |  +-------+  |  |  +-------+  |  |  +-------+  |               |
|  |  | Probe |  |  |  | Probe |  |  |  | Probe |  |               |
|  |  | Agent |  |  |  | Agent |  |  |  | Agent |  |               |
|  |  +---+---+  |  |  +---+---+  |  |  +---+---+  |               |
|  |      |       |  |      |       |  |      |       |               |
|  |  +---+---+  |  |  +---+---+  |  |  +---+---+  |               |
|  |  | Inf.  |  |  |  | Train |  |  |  | Inf.  |  |               |
|  |  | Mon.  |  |  |  | Mon.  |  |  |  | Mon.  |  |               |
|  |  +---+---+  |  |  +---+---+  |  |  +---+---+  |               |
|  +------+------+  +------+------+  +------+------+               |
|         |                |                |                       |
+---------|----------------|----------------|-------+               |
          |                |                |                       |
     +----v----------------v----------------v----+                  |
     |          Correlation Engine (Rust)         |                  |
     |  - Temporal & spatial anomaly detection    |                  |
     |  - Fleet-wide pattern recognition          |                  |
     |  - GPU quarantine decisions                |                  |
     +--------------------+----------------------+                  |
                          |                                         |
     +--------------------v----------------------+                  |
     |           Audit Ledger (Rust)             |                  |
     |  - Tamper-evident hash chain              |                  |
     |  - Compliance & forensics                 |                  |
     |  - Full event history                     |                  |
     +-------------------------------------------+                  |
                                                                    |
     +-------------------------------------------+                  |
     |           Dashboard (React)               |                  |
     |  - Real-time fleet health                 |                  |
     |  - SDC event timeline                     |                  |
     |  - GPU drill-down                         |                  |
     +-------------------------------------------+                  |
```

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| **[Probe Agent](probe-agent/)** | CUDA/C++ | Runs deterministic GPU micro-benchmarks (FMA, tensor core, memory, transcendental, AES) and compares results against golden answers. Detects SDC at the hardware level. |
| **[Inference Monitor](inference-monitor/)** | Python | Sidecar that samples inference requests and validates output distributions against statistical baselines. Catches SDC that manifests as output anomalies. |
| **[Training Monitor](training-monitor/)** | Python | Hooks into training loops to monitor gradient magnitudes, loss trajectories, and cross-GPU consistency. Detects SDC during distributed training. |
| **[Correlation Engine](correlation-engine/)** | Rust | Ingests events from all agents and monitors, applies temporal and spatial correlation, detects fleet-wide failure patterns, and issues quarantine decisions. |
| **[Audit Ledger](audit-ledger/)** | Rust | Tamper-evident append-only log of all SDC events, operator actions, and configuration changes. Uses hash chains for integrity verification. |
| **[Dashboard](dashboard/)** | React/TypeScript | Real-time visualization of fleet health, SDC events, and GPU drill-down. |
| **[SDK](sdk/)** | Python | Client libraries for integrating SENTINEL into ML training and inference pipelines. |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 12.3+ (for probe agent)
- Kubernetes 1.28+ (for production deployment)

### Local Development (Docker Compose)

```bash
# Clone the repository
git clone https://github.com/sentinel-sdc/sentinel.git
cd sentinel

# Start all services
docker compose -f deploy/docker-compose.yml up -d

# Check service health
curl http://localhost:8080/health    # Correlation Engine
curl http://localhost:8083/health    # Audit Ledger

# View the dashboard
open http://localhost:3000
```

### Running the Probe Agent

```bash
# Build
cd probe-agent
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
cmake --build . --parallel

# Run with default configuration
./sentinel-probe-agent --config /etc/sentinel/probe-agent.yaml
```

### Running SDC Injection Tests

```bash
# Build the SDC injector
cd tools/sdc-injector
mkdir build && cd build
cmake ..
cmake --build . --parallel

# Run self-test
./sdc-injector --enable-injection selftest

# Run the full test harness
cd ../src
python harness.py --sentinel-api http://localhost:8080 --scenarios all
```

### Generating Golden Answers

```bash
cd tools/golden-answer-generator
pip install mpmath
python generate.py --all --output-dir golden/
python verify.py --golden-dir golden/
```

## Production Deployment (Kubernetes)

```bash
# Add the Helm repository
helm repo add sentinel oci://ghcr.io/sentinel-sdc/sentinel/helm

# Install SENTINEL
helm install sentinel sentinel/sentinel \
  --namespace sentinel-system \
  --create-namespace \
  --values deploy/helm/values-production.yaml

# Verify deployment
kubectl get pods -n sentinel-system
```

## Building from Source

### Probe Agent (CUDA/C++)

```bash
cd probe-agent
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Correlation Engine (Rust)

```bash
cd correlation-engine
cargo build --release
cargo test
```

### Audit Ledger (Rust)

```bash
cd audit-ledger
cargo build --release
cargo test
```

### Python Components

```bash
# Inference Monitor
cd inference-monitor
pip install -e ".[dev]"
pytest tests/

# Training Monitor
cd training-monitor
pip install -e ".[dev]"
pytest tests/
```

## Benchmarks

SENTINEL is designed for minimal overhead on production GPU workloads.

```bash
# Measure probe agent overhead
python benchmarks/overhead_measurement/probe_overhead.py --all-schedules

# Measure inference monitor overhead
python benchmarks/overhead_measurement/inference_monitor_overhead.py

# Load test the correlation engine
python benchmarks/scalability/correlation_engine_load.py --endpoint localhost:50051

# Benchmark audit ledger throughput
python benchmarks/scalability/audit_ledger_throughput.py --endpoint localhost:50052
```

## Project Structure

```
sentinel/
+-- probe-agent/              # CUDA/C++ GPU probe agent
+-- inference-monitor/        # Python inference sidecar
+-- training-monitor/         # Python training hook
+-- correlation-engine/       # Rust event correlation
+-- audit-ledger/             # Rust tamper-evident log
+-- dashboard/                # React frontend
+-- sdk/                      # Python client SDK
+-- proto/                    # gRPC/Protobuf definitions
+-- config/                   # Default configurations
+-- deploy/                   # Helm charts, docker-compose
+-- tools/
|   +-- sdc-injector/         # Controlled SDC injection
|   +-- golden-answer-generator/ # Reference value generation
|   +-- fleet-simulator/      # Fleet simulation for testing
+-- benchmarks/               # Performance benchmarks
+-- .github/                  # CI/CD workflows
```

## Status

SENTINEL is in **pre-release alpha** (v0.1.0-alpha). This means:

- **APIs are unstable.** gRPC service definitions, SDK interfaces, and configuration schemas may change without notice between versions.
- **Not production-hardened.** While the architecture is designed for production use, this release has not undergone the field testing, performance validation, or security auditing required for production deployments.
- **Community feedback welcome.** We are actively seeking feedback on the detection methodology, system architecture, and API design. Please open issues or discussions on GitHub.
- **Contributions encouraged.** See the [Contributing](#contributing) section below.

Planned milestones toward a stable release:

| Milestone | Target | Status |
|-----------|--------|--------|
| v0.1.0-alpha | Q1 2026 | Current |
| v0.2.0-alpha (probe engine validated on H100/A100) | Q2 2026 | Planned |
| v0.3.0-beta (full pipeline E2E tested) | Q3 2026 | Planned |
| v0.4.0-beta (field trial on partner cluster) | Q4 2026 | Planned |
| v1.0.0 (stable release) | Q1 2027 | Planned |

## Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Architecture Overview](docs/architecture.md) -- System design, component interactions, and data flow
- [Deployment Guide](docs/deployment-guide.md) -- Production deployment on Kubernetes and bare metal
- [Operator Runbook](docs/operator-runbook.md) -- Day-to-day operations, alert triage, quarantine management
- [API Reference](docs/api-reference.md) -- gRPC and REST API documentation
- [SDK Guide (Python)](docs/sdk-python.md) -- Python SDK installation, usage, and examples
- [SDK Guide (Go)](docs/sdk-go.md) -- Go SDK installation, usage, and examples
- [SDC Primer](docs/sdc-primer.md) -- Background on silent data corruption
- [Calibration Guide](docs/calibration-guide.md) -- Threshold tuning and configuration
- [Probe Development](docs/probe-development.md) -- Writing custom probe kernels
- [Compliance](docs/compliance/) -- SOC 2 and ISO 27001 control mappings

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up your development environment, running tests, and submitting pull requests.

## Security

For reporting security vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

SENTINEL is open-source software licensed under the **[Apache License, Version 2.0](LICENSE)**.

```
Copyright 2025-2026 SENTINEL Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
