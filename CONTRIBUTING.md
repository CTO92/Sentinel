# Contributing to SENTINEL

Thank you for your interest in contributing to SENTINEL. This document provides guidelines and instructions for contributing.

## Code of Conduct

All contributors are expected to adhere to professional standards of conduct. Be respectful, constructive, and collaborative in all interactions. Harassment, discrimination, and disruptive behavior will not be tolerated.

## Getting Started

### Development Environment

**Prerequisites:**
- Git
- Docker and Docker Compose
- For Probe Agent: NVIDIA GPU, CUDA Toolkit 12.3+, CMake 3.24+, GCC/G++ 11+
- For Python components: Python 3.11+
- For Rust components: Rust 1.75+ (install via [rustup](https://rustup.rs))
- For Dashboard: Node.js 20+

### Clone and Setup

```bash
git clone https://github.com/sentinel-sdc/sentinel.git
cd sentinel
```

### Component-Specific Setup

**Probe Agent (CUDA/C++):**
```bash
cd probe-agent
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

**Inference Monitor (Python):**
```bash
cd inference-monitor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

**Training Monitor (Python):**
```bash
cd training-monitor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

**Correlation Engine (Rust):**
```bash
cd correlation-engine
cargo build
cargo test
```

**Audit Ledger (Rust):**
```bash
cd audit-ledger
cargo build
cargo test
```

## Development Workflow

### Branching

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Use descriptive branch names: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`.

### Making Changes

1. Write code following the style guidelines for each language (see below).
2. Add or update tests for your changes.
3. Ensure all existing tests pass.
4. Update documentation if your changes affect user-facing behavior.

### Commit Messages

Use clear, descriptive commit messages:

```
component: short summary (imperative mood)

Longer description of what and why, if needed. Wrap at 72 characters.
Reference any related issues.

Fixes #123
```

Examples:
- `probe-agent: add tensor core probe for FP8 formats`
- `correlation-engine: fix temporal window race condition`
- `docs: update deployment guide for Kubernetes 1.29`

### Pull Request Process

1. Push your branch and open a Pull Request against `main`.
2. Fill in the PR template with a summary, test plan, and any relevant context.
3. Ensure CI passes. All required checks must be green before merge.
4. Request review from the appropriate code owners (see `.github/CODEOWNERS`).
5. Address review feedback with additional commits (do not force-push during review).
6. A maintainer will merge the PR once approved.

## Testing Requirements

### All Components

- Unit tests are required for new functionality.
- Integration tests are required for cross-component interactions.
- Bug fixes must include a regression test.

### Probe Agent (C++/CUDA)

- Use Google Test for unit tests.
- GPU-specific tests must be tagged with the `gpu` label in CTest.
- Run `clang-tidy` before submitting.

### Python Components

- Use `pytest` for all tests.
- Mark GPU-dependent tests with `@pytest.mark.gpu`.
- Maintain type annotations; run `mypy` with `--strict` where possible.
- Format with `ruff format` and lint with `ruff check`.
- Target 80%+ code coverage.

### Rust Components

- Use `cargo test` for unit and integration tests.
- Run `cargo fmt` and `cargo clippy` before submitting.
- Integration tests requiring external services (PostgreSQL, ScyllaDB) should be `#[ignore]` by default.
- Run benchmarks with `cargo bench` and verify no regressions.

## Style Guidelines

### C++ / CUDA

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with these exceptions:
  - 4-space indentation
  - 100-character line limit
- Use `clang-format` with the project `.clang-format` configuration.

### Python

- Follow PEP 8 via `ruff`.
- Use type annotations for all public functions.
- Use `from __future__ import annotations` for modern annotation syntax.
- Docstrings for all public modules, classes, and functions (Google style).

### Rust

- Follow standard Rust formatting (`cargo fmt`).
- All public items must have documentation comments (`///`).
- Use `clippy` lints at the default warning level.

## Reporting Issues

- **Bugs**: Use the [Bug Report](https://github.com/sentinel-sdc/sentinel/issues/new?template=bug_report.md) template.
- **Features**: Use the [Feature Request](https://github.com/sentinel-sdc/sentinel/issues/new?template=feature_request.md) template.
- **SDC Incidents**: Use the [SDC Incident](https://github.com/sentinel-sdc/sentinel/issues/new?template=sdc_incident.md) template.

## Security

Do not open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

By contributing to SENTINEL, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
