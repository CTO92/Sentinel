# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.x.x   | Current development branch |

Security patches are applied to the latest release only. We recommend always running the most recent version.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in SENTINEL, please report it responsibly:

1. **Email**: Send a detailed report to **security@sentinel-sdc.dev**
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Affected component(s) and version(s)
   - Potential impact assessment
   - Suggested fix, if any

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours.
- **Assessment**: We will assess the vulnerability and determine severity within 5 business days.
- **Resolution**: We aim to release a fix within 30 days for critical vulnerabilities, 90 days for others.
- **Disclosure**: We will coordinate disclosure timing with you. We follow a 90-day disclosure policy.
- **Credit**: We will credit reporters in the release notes (unless you prefer to remain anonymous).

## Security Considerations for SENTINEL

SENTINEL handles sensitive information related to GPU fleet health and model integrity. Key security areas:

### Audit Ledger Integrity

The audit ledger uses SHA-256 hash chains to ensure tamper evidence. Any modification to historical records will break the chain and be detected during verification. The ledger should be treated as a compliance-critical data store.

### gRPC Communication

All inter-component gRPC communication should use mTLS in production deployments. The default development configuration uses insecure channels for convenience.

### Probe Agent Privileges

The probe agent requires elevated GPU access (CUDA driver API) to run diagnostic kernels. It should run with the minimum necessary privileges. The SDC injector tool requires an explicit `--enable-injection` flag and must never be deployed in production.

### Container Security

- All container images are based on minimal base images.
- Images run as non-root users.
- Weekly Trivy scans check for known vulnerabilities.
- Dependency audits run for all language ecosystems (cargo-audit, pip-audit, CodeQL).

### Configuration Secrets

- Database credentials, API keys, and TLS certificates should be provided via Kubernetes Secrets or a secrets manager.
- Never commit secrets to the repository.
- The `.gitignore` is configured to exclude common secret file patterns.

## Security-Related Configuration

### Recommended Production Settings

```yaml
# TLS for gRPC
grpc:
  tls:
    enabled: true
    cert_file: /etc/sentinel/tls/tls.crt
    key_file: /etc/sentinel/tls/tls.key
    ca_file: /etc/sentinel/tls/ca.crt

# Audit ledger
audit:
  retention_days: 365
  chain_verification_interval: 3600

# Network policies
network:
  restrict_egress: true
  allowed_namespaces:
    - sentinel-system
    - monitoring
```

## Dependencies

We monitor dependencies for known vulnerabilities using:
- **Rust**: `cargo audit` (weekly CI)
- **Python**: `pip-audit` (weekly CI)
- **Containers**: Trivy (weekly CI)
- **Code**: GitHub CodeQL (weekly CI)
