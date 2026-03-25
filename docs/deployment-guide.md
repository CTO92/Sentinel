# SENTINEL Deployment Guide

> **Version:** 0.1.0-alpha | **Status:** Pre-release | **Last Updated:** 2026-03-24

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development (Docker Compose)](#local-development-docker-compose)
3. [Kubernetes Production Deployment](#kubernetes-production-deployment)
4. [Bare Metal Deployment](#bare-metal-deployment)
5. [Configuration Reference](#configuration-reference)
6. [Upgrading](#upgrading)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Probe Agent nodes** | NVIDIA GPU with CUDA 12.3+, 1 CPU core, 512 MB RAM available | 2 CPU cores, 1 GB RAM per GPU |
| **Correlation Engine** | 2 CPU cores, 2 GB RAM | 4+ CPU cores, 4-8 GB RAM (scales with fleet size) |
| **Audit Ledger** | 1 CPU core, 512 MB RAM | 2 CPU cores, 2 GB RAM |
| **PostgreSQL** | 2 CPU cores, 2 GB RAM, 50 GB SSD | 4 CPU cores, 8 GB RAM, 200 GB NVMe |
| **ScyllaDB** | 2 CPU cores, 4 GB RAM, 100 GB SSD | 4+ CPU cores, 8+ GB RAM, 500 GB NVMe |
| **Redis** | 1 CPU core, 512 MB RAM | 1 CPU core, 2 GB RAM |

### Software Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Docker | 24.0+ | Container runtime (local dev) |
| Docker Compose | 2.20+ | Local multi-service orchestration |
| Kubernetes | 1.28+ | Production orchestration |
| Helm | 3.14+ | Kubernetes package management (optional) |
| NVIDIA Container Toolkit | 1.14+ | GPU access in containers |
| NVIDIA Driver | 535+ | GPU driver with CUDA 12.3+ support |
| CUDA Toolkit | 12.3+ | Probe agent compilation (if building from source) |
| Rust | 1.75+ | Correlation engine / audit ledger compilation (if building from source) |
| Python | 3.10+ | Inference and training monitors |
| CMake | 3.25+ | Probe agent build system (if building from source) |

### Network Requirements

| Connection | Protocol | Port | Direction |
|------------|----------|------|-----------|
| Probe Agent -> Correlation Engine | gRPC (HTTP/2) | 50051 | Outbound from GPU nodes |
| Inference/Training Monitor -> Correlation Engine | gRPC (HTTP/2) | 50051 | Outbound from GPU nodes |
| Correlation Engine -> Audit Ledger | gRPC (HTTP/2) | 50052 | Internal |
| Correlation Engine -> PostgreSQL | TCP | 5432 | Internal |
| Correlation Engine -> ScyllaDB | TCP (CQL) | 9042 | Internal |
| Correlation Engine -> Redis | TCP | 6379 | Internal |
| Audit Ledger -> ScyllaDB | TCP (CQL) | 9042 | Internal |
| Prometheus -> Correlation Engine metrics | HTTP | 9090 | Internal |
| Prometheus -> Audit Ledger metrics | HTTP | 9091 | Internal |
| Dashboard -> Correlation Engine | gRPC-Web / HTTP | 50051/8080 | Internal |
| Dashboard -> Audit Ledger | gRPC-Web / HTTP | 50052/8083 | Internal |
| Alerting -> External (Slack, PagerDuty, SMTP) | HTTPS/SMTP | 443/587 | Outbound |

Minimum bandwidth: 1 Mbps per 1,000 monitored GPUs for SENTINEL telemetry traffic. This is negligible relative to typical GPU cluster network traffic.

---

## Local Development (Docker Compose)

The Docker Compose setup starts all SENTINEL services locally for development and testing. GPU probe execution requires an NVIDIA GPU and the NVIDIA Container Toolkit.

### Starting All Services

```bash
# Clone the repository
git clone https://github.com/sentinel-sdc/sentinel.git
cd sentinel

# Start the infrastructure (databases, observability) and SENTINEL services
docker compose -f deploy/docker-compose.yml up -d

# Watch startup logs
docker compose -f deploy/docker-compose.yml logs -f

# Verify all services are healthy
docker compose -f deploy/docker-compose.yml ps
```

Expected healthy services:

| Service | Container Name | Ports |
|---------|---------------|-------|
| Correlation Engine | sentinel-correlation-engine | 50051 (gRPC), 9090 (metrics) |
| Audit Ledger | sentinel-audit-ledger | 50052 (gRPC), 9091 (metrics) |
| PostgreSQL | sentinel-postgres | 5432 |
| ScyllaDB | sentinel-scylladb | 9042, 9180 |
| Redis | sentinel-redis | 6379 |
| Prometheus | sentinel-prometheus | 9092 |
| Grafana | sentinel-grafana | 3000 |

**Note:** ScyllaDB takes 60-90 seconds to become healthy. The correlation engine and audit ledger will wait for it automatically via Docker Compose health check dependencies.

### Connecting the Probe Agent

The probe agent runs outside Docker Compose (it needs direct GPU access):

```bash
# Build the probe agent
cd probe-agent
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build --parallel

# Run with local development configuration
# TLS is disabled in docker-compose mode
./build/sentinel-probe-agent \
  --config ../config/sentinel.default.yaml \
  --override correlation_engine_endpoint=localhost:50051 \
  --override tls.enabled=false
```

Verify the probe agent is connected:

```bash
# Check correlation engine logs for agent registration
docker compose -f deploy/docker-compose.yml logs correlation-engine | grep "agent registered"

# Check probe results are flowing
curl -s http://localhost:9090/metrics | grep sentinel_probe_results_total
```

### Running the Dashboard

```bash
# The dashboard is available if included in docker-compose, or run standalone:
cd dashboard
npm install
npm run dev
# Open http://localhost:3000 (or the Grafana instance on the same port)
```

For the default Docker Compose setup, Grafana is available at http://localhost:3000 with credentials `admin`/`sentinel`.

### Running Tests

```bash
# Run integration tests (requires running services)
docker compose -f deploy/docker-compose.test.yml up --build --abort-on-container-exit

# Run unit tests for individual components
cd probe-agent && ctest --test-dir build --output-on-failure
cd correlation-engine && cargo test
cd audit-ledger && cargo test
cd inference-monitor && pip install -e ".[dev]" && pytest tests/
cd training-monitor && pip install -e ".[dev]" && pytest tests/
```

### Stopping Services

```bash
# Stop all services (preserves data volumes)
docker compose -f deploy/docker-compose.yml down

# Stop all services and remove data volumes
docker compose -f deploy/docker-compose.yml down -v
```

---

## Kubernetes Production Deployment

### Overview

The production Kubernetes deployment uses the following resource types:

| Component | K8s Resource | Scaling |
|-----------|-------------|---------|
| Correlation Engine | Deployment + HPA | 1-12 replicas based on fleet size |
| Audit Ledger | Deployment (single-writer) + read replicas | 1 writer + N readers |
| Probe Agent | DaemonSet (GPU nodes) | 1 per GPU node (automatic) |
| Inference Monitor | Sidecar injection | 1 per inference pod |
| Training Monitor | Library (in-process) | N/A |

### Step 1: Namespace and RBAC Setup

```bash
# Apply namespace and RBAC resources
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/rbac.yaml
```

The namespace configuration creates a `sentinel` namespace with appropriate labels. The RBAC configuration creates:

- `sentinel-probe-agent` ServiceAccount: Used by the DaemonSet. Has permissions to read node labels and GPU topology.
- `sentinel-correlation-engine` ServiceAccount: Used by the Correlation Engine. Has permissions to manage ConfigMaps (for dynamic config), read/write Secrets (for TLS certs).
- `sentinel-audit-ledger` ServiceAccount: Used by the Audit Ledger. Minimal permissions (read Secrets for TLS certs only).

### Step 2: TLS Certificate Management

SENTINEL requires TLS certificates for all gRPC communication in production.

**Option A: cert-manager (recommended)**

```yaml
# cert-manager Certificate resource for SENTINEL CA
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: sentinel-ca-issuer
  namespace: sentinel
spec:
  selfSigned: {}
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: sentinel-ca
  namespace: sentinel
spec:
  isCA: true
  commonName: sentinel-ca
  secretName: sentinel-ca-tls
  issuerRef:
    name: sentinel-ca-issuer
    kind: Issuer
---
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: sentinel-issuer
  namespace: sentinel
spec:
  ca:
    secretName: sentinel-ca-tls
---
# Issue certificates for each component
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: sentinel-correlation-engine-tls
  namespace: sentinel
spec:
  secretName: correlation-engine-tls
  issuerRef:
    name: sentinel-issuer
    kind: Issuer
  commonName: correlation-engine
  dnsNames:
    - correlation-engine
    - correlation-engine.sentinel.svc
    - correlation-engine.sentinel.svc.cluster.local
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: sentinel-audit-ledger-tls
  namespace: sentinel
spec:
  secretName: audit-ledger-tls
  issuerRef:
    name: sentinel-issuer
    kind: Issuer
  commonName: audit-ledger
  dnsNames:
    - audit-ledger
    - audit-ledger.sentinel.svc
    - audit-ledger.sentinel.svc.cluster.local
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: sentinel-probe-agent-tls
  namespace: sentinel
spec:
  secretName: probe-agent-tls
  issuerRef:
    name: sentinel-issuer
    kind: Issuer
  commonName: probe-agent
```

**Option B: Manual certificate generation**

```bash
# Generate CA
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt -subj "/CN=sentinel-ca"

# Generate component certificates
for component in correlation-engine audit-ledger probe-agent; do
  openssl genrsa -out ${component}.key 2048
  openssl req -new -key ${component}.key -out ${component}.csr \
    -subj "/CN=${component}"
  openssl x509 -req -days 365 -in ${component}.csr \
    -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out ${component}.crt
done

# Create Kubernetes secrets
kubectl create secret tls correlation-engine-tls \
  --cert=correlation-engine.crt --key=correlation-engine.key -n sentinel
kubectl create secret tls audit-ledger-tls \
  --cert=audit-ledger.crt --key=audit-ledger.key -n sentinel
kubectl create secret tls probe-agent-tls \
  --cert=probe-agent.crt --key=probe-agent.key -n sentinel
kubectl create secret generic sentinel-ca-tls \
  --from-file=ca.crt=ca.crt -n sentinel
```

### Step 3: Database Setup

#### PostgreSQL

Use a managed PostgreSQL service (e.g., Amazon RDS, Google Cloud SQL, Azure Database for PostgreSQL) for production. Alternatively, deploy via Helm:

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami

helm install sentinel-postgres bitnami/postgresql \
  --namespace sentinel \
  --set auth.username=sentinel \
  --set auth.password=<STRONG_PASSWORD> \
  --set auth.database=sentinel \
  --set primary.persistence.size=200Gi \
  --set primary.persistence.storageClass=fast-ssd \
  --set primary.resources.requests.memory=4Gi \
  --set primary.resources.requests.cpu=2 \
  --set primary.resources.limits.memory=8Gi \
  --set primary.resources.limits.cpu=4 \
  --set tls.enabled=true
```

#### ScyllaDB

Use the Scylla Operator for Kubernetes:

```bash
# Install the Scylla Operator
kubectl apply -f https://raw.githubusercontent.com/scylladb/scylla-operator/master/deploy/operator.yaml

# Deploy a ScyllaDB cluster
cat <<EOF | kubectl apply -f -
apiVersion: scylla.scylladb.com/v1
kind: ScyllaCluster
metadata:
  name: sentinel-scylla
  namespace: sentinel
spec:
  version: "5.4"
  datacenter:
    name: dc1
    racks:
      - name: rack1
        members: 3
        storage:
          capacity: 500Gi
          storageClassName: fast-ssd
        resources:
          requests:
            cpu: 4
            memory: 8Gi
          limits:
            cpu: 8
            memory: 16Gi
EOF
```

#### Redis

```bash
helm install sentinel-redis bitnami/redis \
  --namespace sentinel \
  --set auth.enabled=true \
  --set auth.password=<STRONG_PASSWORD> \
  --set master.persistence.size=10Gi \
  --set master.resources.requests.memory=1Gi \
  --set master.resources.limits.memory=2Gi \
  --set replica.replicaCount=0
```

### Step 4: Deploy the Correlation Engine

```yaml
# deploy/kubernetes/correlation-engine-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: correlation-engine
  namespace: sentinel
  labels:
    app.kubernetes.io/name: correlation-engine
    app.kubernetes.io/part-of: sentinel-sdc
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: correlation-engine
  template:
    metadata:
      labels:
        app.kubernetes.io/name: correlation-engine
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: sentinel-correlation-engine
      containers:
        - name: correlation-engine
          image: ghcr.io/sentinel-sdc/correlation-engine:0.1.0-alpha
          ports:
            - name: grpc
              containerPort: 50051
            - name: metrics
              containerPort: 9090
          env:
            - name: SENTINEL_CORRELATION_ENGINE_LISTEN_ADDRESS
              value: "0.0.0.0:50051"
            - name: SENTINEL_CORRELATION_ENGINE_DATABASE_POSTGRES_URL
              valueFrom:
                secretKeyRef:
                  name: sentinel-db-credentials
                  key: postgres-url
            - name: SENTINEL_CORRELATION_ENGINE_SCYLLA_CONTACT_POINTS
              value: "sentinel-scylla-client.sentinel.svc:9042"
            - name: SENTINEL_CORRELATION_ENGINE_REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: sentinel-db-credentials
                  key: redis-url
            - name: SENTINEL_CORRELATION_ENGINE_TLS_ENABLED
              value: "true"
            - name: SENTINEL_CORRELATION_ENGINE_TLS_CERT_FILE
              value: "/etc/sentinel/certs/tls.crt"
            - name: SENTINEL_CORRELATION_ENGINE_TLS_KEY_FILE
              value: "/etc/sentinel/certs/tls.key"
            - name: SENTINEL_CORRELATION_ENGINE_TLS_CA_FILE
              value: "/etc/sentinel/certs/ca.crt"
          volumeMounts:
            - name: tls-certs
              mountPath: /etc/sentinel/certs
              readOnly: true
            - name: config
              mountPath: /etc/sentinel/config
              readOnly: true
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 8Gi
          readinessProbe:
            exec:
              command: ["/usr/local/bin/grpc_health_probe", "-addr=:50051"]
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            exec:
              command: ["/usr/local/bin/grpc_health_probe", "-addr=:50051"]
            initialDelaySeconds: 30
            periodSeconds: 15
      volumes:
        - name: tls-certs
          projected:
            sources:
              - secret:
                  name: correlation-engine-tls
              - secret:
                  name: sentinel-ca-tls
        - name: config
          configMap:
            name: sentinel-config
---
apiVersion: v1
kind: Service
metadata:
  name: correlation-engine
  namespace: sentinel
spec:
  selector:
    app.kubernetes.io/name: correlation-engine
  ports:
    - name: grpc
      port: 50051
      targetPort: 50051
    - name: metrics
      port: 9090
      targetPort: 9090
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: correlation-engine
  namespace: sentinel
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: correlation-engine
  minReplicas: 2
  maxReplicas: 12
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: sentinel_grpc_active_streams
        target:
          type: AverageValue
          averageValue: "500"
```

### Step 5: Deploy the Audit Ledger

The audit ledger runs as a single-writer deployment with optional read replicas.

```yaml
# Writer deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audit-ledger-writer
  namespace: sentinel
  labels:
    app.kubernetes.io/name: audit-ledger
    audit-ledger/role: writer
spec:
  replicas: 1  # MUST be 1 for hash chain integrity
  strategy:
    type: Recreate  # Never run two writers simultaneously
  selector:
    matchLabels:
      app.kubernetes.io/name: audit-ledger
      audit-ledger/role: writer
  template:
    metadata:
      labels:
        app.kubernetes.io/name: audit-ledger
        audit-ledger/role: writer
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9091"
    spec:
      serviceAccountName: sentinel-audit-ledger
      containers:
        - name: audit-ledger
          image: ghcr.io/sentinel-sdc/audit-ledger:0.1.0-alpha
          args: ["--mode", "writer"]
          ports:
            - name: grpc
              containerPort: 50052
            - name: metrics
              containerPort: 9091
          env:
            - name: SENTINEL_AUDIT_LEDGER_LISTEN_ADDRESS
              value: "0.0.0.0:50052"
            - name: SENTINEL_AUDIT_LEDGER_STORAGE_SCYLLA_CONTACT_POINTS
              value: "sentinel-scylla-client.sentinel.svc:9042"
            - name: SENTINEL_AUDIT_LEDGER_TLS_ENABLED
              value: "true"
            - name: SENTINEL_AUDIT_LEDGER_TLS_CERT_FILE
              value: "/etc/sentinel/certs/tls.crt"
            - name: SENTINEL_AUDIT_LEDGER_TLS_KEY_FILE
              value: "/etc/sentinel/certs/tls.key"
            - name: SENTINEL_AUDIT_LEDGER_TLS_CA_FILE
              value: "/etc/sentinel/certs/ca.crt"
          volumeMounts:
            - name: tls-certs
              mountPath: /etc/sentinel/certs
              readOnly: true
            - name: config
              mountPath: /etc/sentinel/config
              readOnly: true
          resources:
            requests:
              cpu: "1"
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 2Gi
          readinessProbe:
            exec:
              command: ["/usr/local/bin/grpc_health_probe", "-addr=:50052"]
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            exec:
              command: ["/usr/local/bin/grpc_health_probe", "-addr=:50052"]
            initialDelaySeconds: 30
            periodSeconds: 15
      volumes:
        - name: tls-certs
          projected:
            sources:
              - secret:
                  name: audit-ledger-tls
              - secret:
                  name: sentinel-ca-tls
        - name: config
          configMap:
            name: sentinel-config
---
# Read replica deployment (optional, for query scaling)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audit-ledger-reader
  namespace: sentinel
  labels:
    app.kubernetes.io/name: audit-ledger
    audit-ledger/role: reader
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: audit-ledger
      audit-ledger/role: reader
  template:
    metadata:
      labels:
        app.kubernetes.io/name: audit-ledger
        audit-ledger/role: reader
    spec:
      serviceAccountName: sentinel-audit-ledger
      containers:
        - name: audit-ledger
          image: ghcr.io/sentinel-sdc/audit-ledger:0.1.0-alpha
          args: ["--mode", "reader"]
          ports:
            - name: grpc
              containerPort: 50052
          # ... (same env/volumes as writer, omitted for brevity)
          resources:
            requests:
              cpu: "500m"
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: audit-ledger
  namespace: sentinel
spec:
  selector:
    app.kubernetes.io/name: audit-ledger
    audit-ledger/role: writer
  ports:
    - name: grpc
      port: 50052
      targetPort: 50052
    - name: metrics
      port: 9091
      targetPort: 9091
---
apiVersion: v1
kind: Service
metadata:
  name: audit-ledger-reader
  namespace: sentinel
spec:
  selector:
    app.kubernetes.io/name: audit-ledger
    audit-ledger/role: reader
  ports:
    - name: grpc
      port: 50052
      targetPort: 50052
```

### Step 6: Deploy Probe Agents (DaemonSet)

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: sentinel-probe-agent
  namespace: sentinel
  labels:
    app.kubernetes.io/name: probe-agent
    app.kubernetes.io/part-of: sentinel-sdc
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: probe-agent
  template:
    metadata:
      labels:
        app.kubernetes.io/name: probe-agent
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9092"
    spec:
      serviceAccountName: sentinel-probe-agent
      nodeSelector:
        # Only schedule on GPU nodes
        nvidia.com/gpu.present: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: probe-agent
          image: ghcr.io/sentinel-sdc/probe-agent:0.1.0-alpha
          securityContext:
            privileged: false
            capabilities:
              add:
                - SYS_ADMIN  # Required for NVML access
          env:
            - name: SENTINEL_PROBE_AGENT_CORRELATION_ENGINE_ENDPOINT
              value: "correlation-engine.sentinel.svc:50051"
            - name: SENTINEL_PROBE_AGENT_TLS_ENABLED
              value: "true"
            - name: SENTINEL_PROBE_AGENT_TLS_CERT_FILE
              value: "/etc/sentinel/certs/tls.crt"
            - name: SENTINEL_PROBE_AGENT_TLS_KEY_FILE
              value: "/etc/sentinel/certs/tls.key"
            - name: SENTINEL_PROBE_AGENT_TLS_CA_FILE
              value: "/etc/sentinel/certs/ca.crt"
            - name: SENTINEL_PROBE_AGENT_HMAC_KEY
              valueFrom:
                secretKeyRef:
                  name: sentinel-hmac-key
                  key: key
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
          volumeMounts:
            - name: tls-certs
              mountPath: /etc/sentinel/certs
              readOnly: true
            - name: config
              mountPath: /etc/sentinel/config
              readOnly: true
          resources:
            requests:
              cpu: "500m"
              memory: 256Mi
              nvidia.com/gpu: "0"  # Does NOT consume GPU resource quota
            limits:
              cpu: "2"
              memory: 1Gi
      volumes:
        - name: tls-certs
          projected:
            sources:
              - secret:
                  name: probe-agent-tls
              - secret:
                  name: sentinel-ca-tls
        - name: config
          configMap:
            name: sentinel-config
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: "25%"
```

**Important:** The probe agent requests `nvidia.com/gpu: "0"` -- it accesses GPUs via NVML and CUDA runtime without consuming Kubernetes GPU resource quotas. This allows the probe agent to coexist with production GPU workloads.

### Step 7: Deploy Inference Monitor (Sidecar Injection)

The inference monitor runs as a sidecar container alongside inference serving pods. It can be injected via a mutating webhook or manually added to pod specs.

**Manual sidecar addition:**

```yaml
# Add to your inference pod spec
containers:
  - name: inference-server
    image: your-inference-image:latest
    # ... your existing container spec
  - name: sentinel-inference-monitor
    image: ghcr.io/sentinel-sdc/inference-monitor:0.1.0-alpha
    env:
      - name: SENTINEL_INFERENCE_MONITOR_CORRELATION_ENGINE_ENDPOINT
        value: "correlation-engine.sentinel.svc:50051"
      - name: SENTINEL_INFERENCE_MONITOR_SAMPLING_RATE
        value: "0.01"
      - name: SENTINEL_INFERENCE_MONITOR_FRAMEWORK
        value: "vllm"  # or "trtllm", "triton", "generic"
    resources:
      requests:
        cpu: "100m"
        memory: 256Mi
      limits:
        cpu: "500m"
        memory: 512Mi
```

**Library integration (preferred for vLLM and TRT-LLM):**

```python
# In your inference server code
from sentinel_inference import InferenceMonitor

monitor = InferenceMonitor.from_config("/etc/sentinel/config/sentinel.yaml")
monitor.attach_vllm(engine)  # or attach_trtllm(), attach_triton()
```

### Step 8: Deploy Training Monitor (Library Integration)

The training monitor is a Python library integrated directly into training scripts:

```bash
pip install sentinel-training
```

```python
import sentinel_training

monitor = sentinel_training.pytorch.TrainingMonitor(
    config_path="/etc/sentinel/config/sentinel.yaml"
)
monitor.attach(model, optimizer)

# Training proceeds normally
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

# Monitor automatically collects gradient norms, loss values,
# and cross-rank divergence metrics
```

### Step 9: Network Policies

```bash
kubectl apply -f deploy/kubernetes/network-policies.yaml
```

The network policies enforce:

- Probe agents can only reach the correlation engine on port 50051.
- The correlation engine can reach the audit ledger (50052), PostgreSQL (5432), ScyllaDB (9042), Redis (6379).
- The audit ledger can only reach ScyllaDB (9042).
- Deny all other inter-pod traffic within the sentinel namespace.
- Egress to external alerting endpoints (Slack, PagerDuty) is allowed from the correlation engine only.

### Step 10: Monitoring SENTINEL Itself

#### Prometheus Configuration

```yaml
# Add to your Prometheus scrape config
scrape_configs:
  - job_name: sentinel-correlation-engine
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [sentinel]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
        regex: correlation-engine
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        target_label: __address__
        regex: (.+)
        replacement: ${1}

  - job_name: sentinel-audit-ledger
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [sentinel]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
        regex: audit-ledger
        action: keep

  - job_name: sentinel-probe-agent
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [sentinel]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
        regex: probe-agent
        action: keep
```

#### Grafana Dashboards

Import the SENTINEL Grafana dashboards from `deploy/grafana/provisioning/`:

- **Fleet Health Overview:** Fleet-wide GPU health status, quarantine count, SDC event rate.
- **GPU Deep Dive:** Per-GPU reliability score history, probe results, anomaly timeline.
- **Correlation Engine Performance:** gRPC stream counts, event processing latency, buffer utilization.
- **Audit Ledger Status:** Chain integrity status, batch processing rate, storage utilization.

### Resource Requirements and Sizing Guide

| Fleet Size | Corr. Engine Replicas | Corr. Engine CPU/Mem | Audit Writer CPU/Mem | PostgreSQL | ScyllaDB Nodes | Redis |
|-----------|----------------------|---------------------|---------------------|------------|---------------|-------|
| 64 GPUs | 1 | 2 CPU / 2 GB | 1 CPU / 512 MB | 2 CPU / 4 GB | 1 (dev) | 512 MB |
| 256 GPUs | 1 | 4 CPU / 4 GB | 1 CPU / 1 GB | 2 CPU / 4 GB | 3 | 1 GB |
| 1,000 GPUs | 2-3 | 4 CPU / 4 GB each | 2 CPU / 2 GB | 4 CPU / 8 GB | 3 | 2 GB |
| 4,000 GPUs | 4-6 | 4 CPU / 8 GB each | 2 CPU / 2 GB | 4 CPU / 16 GB | 5 | 2 GB |
| 10,000 GPUs | 8-12 | 8 CPU / 16 GB each | 4 CPU / 4 GB | 8 CPU / 32 GB | 7+ | 4 GB |

Storage estimates (per year):

| Fleet Size | ScyllaDB (probe results + audit) | PostgreSQL (state + metadata) |
|-----------|--------------------------------|------------------------------|
| 64 GPUs | ~10 GB | ~1 GB |
| 1,000 GPUs | ~150 GB | ~10 GB |
| 10,000 GPUs | ~1.5 TB | ~100 GB |

---

## Bare Metal Deployment

For environments without Kubernetes, SENTINEL components can be deployed directly as systemd services and binaries.

### Probe Agent (systemd)

```bash
# Install the probe agent binary
sudo cp sentinel-probe-agent /usr/local/bin/
sudo chmod +x /usr/local/bin/sentinel-probe-agent

# Install configuration
sudo mkdir -p /etc/sentinel/config /etc/sentinel/certs
sudo cp config/sentinel.yaml /etc/sentinel/config/
sudo cp config/probe_schedules/ /etc/sentinel/config/probe_schedules/ -r
sudo cp config/thresholds/ /etc/sentinel/config/thresholds/ -r

# Install TLS certificates
sudo cp certs/agent.crt /etc/sentinel/certs/
sudo cp certs/agent.key /etc/sentinel/certs/
sudo cp certs/ca.crt /etc/sentinel/certs/
sudo chmod 600 /etc/sentinel/certs/agent.key

# Create systemd service
sudo tee /etc/systemd/system/sentinel-probe-agent.service > /dev/null <<'EOF'
[Unit]
Description=SENTINEL Probe Agent
Documentation=https://github.com/sentinel-sdc/sentinel
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=simple
User=sentinel
Group=sentinel
ExecStart=/usr/local/bin/sentinel-probe-agent \
  --config /etc/sentinel/config/sentinel.yaml
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/etc/sentinel
PrivateTmp=true

# Environment
Environment=SENTINEL_LOG_LEVEL=info
Environment=SENTINEL_LOG_FORMAT=json
EnvironmentFile=-/etc/sentinel/env

[Install]
WantedBy=multi-user.target
EOF

# Create sentinel user
sudo useradd -r -s /usr/sbin/nologin -G video sentinel

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable sentinel-probe-agent
sudo systemctl start sentinel-probe-agent

# Check status
sudo systemctl status sentinel-probe-agent
sudo journalctl -u sentinel-probe-agent -f
```

### Correlation Engine (systemd)

```bash
sudo cp sentinel-correlation-engine /usr/local/bin/
sudo chmod +x /usr/local/bin/sentinel-correlation-engine

sudo tee /etc/systemd/system/sentinel-correlation-engine.service > /dev/null <<'EOF'
[Unit]
Description=SENTINEL Correlation Engine
Documentation=https://github.com/sentinel-sdc/sentinel
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=sentinel
Group=sentinel
ExecStart=/usr/local/bin/sentinel-correlation-engine \
  --config /etc/sentinel/config/sentinel.yaml
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/etc/sentinel
PrivateTmp=true

Environment=SENTINEL_LOG_LEVEL=info
Environment=SENTINEL_LOG_FORMAT=json
EnvironmentFile=-/etc/sentinel/env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sentinel-correlation-engine
sudo systemctl start sentinel-correlation-engine
```

### Audit Ledger (systemd)

```bash
sudo cp sentinel-audit-ledger /usr/local/bin/
sudo chmod +x /usr/local/bin/sentinel-audit-ledger

sudo tee /etc/systemd/system/sentinel-audit-ledger.service > /dev/null <<'EOF'
[Unit]
Description=SENTINEL Audit Ledger
Documentation=https://github.com/sentinel-sdc/sentinel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=sentinel
Group=sentinel
ExecStart=/usr/local/bin/sentinel-audit-ledger \
  --config /etc/sentinel/config/sentinel.yaml \
  --mode writer
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/etc/sentinel
PrivateTmp=true

Environment=SENTINEL_LOG_LEVEL=info
Environment=SENTINEL_LOG_FORMAT=json
EnvironmentFile=-/etc/sentinel/env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sentinel-audit-ledger
sudo systemctl start sentinel-audit-ledger
```

---

## Configuration Reference

### Master Configuration File

The master configuration file is `sentinel.yaml` (default path: `/etc/sentinel/config/sentinel.yaml`). A fully documented default is provided at `config/sentinel.default.yaml`.

Configuration values can be set via three mechanisms (in priority order):

1. **Dynamic config updates** via the `ConfigService` gRPC stream (highest priority; probe schedules, thresholds).
2. **Environment variables** prefixed with `SENTINEL_` (e.g., `SENTINEL_PROBE_AGENT_OVERHEAD_BUDGET_PCT=1.5`). Nested keys use underscores: `SENTINEL_CORRELATION_ENGINE_BAYESIAN_MODEL_PRIOR_ALPHA=200`.
3. **YAML configuration files** -- base file + optional environment overlay (e.g., `sentinel.production.yaml`).

### Key Configuration Sections

#### Probe Schedules

Probe schedules define what probes run, how often, and with what SM coverage. Three built-in schedules are provided:

| Schedule | File | Overhead Target | Use Case |
|----------|------|----------------|----------|
| `default.yaml` | `config/probe_schedules/default.yaml` | ~1-2% | Standard production |
| `aggressive.yaml` | `config/probe_schedules/aggressive.yaml` | ~3-5% | SUSPECT GPUs or validation periods |
| `low_overhead.yaml` | `config/probe_schedules/low_overhead.yaml` | < 0.5% | Latency-sensitive production |

Custom schedules can define any combination of probe types, periods, and SM coverage fractions.

#### Threshold Tuning

See the [Calibration Guide](calibration-guide.md) for detailed threshold tuning guidance.

Key threshold files:

- `config/thresholds/probe_tolerances.yaml` -- Per-probe-type tolerance modes and ULP limits.
- `config/thresholds/inference_thresholds.yaml` -- EWMA, KL divergence, entropy, and spectral analysis thresholds.
- `config/thresholds/training_thresholds.yaml` -- Gradient norm, loss spike, and cross-rank divergence thresholds.

#### Alerting Configuration

Alert rules are defined in `config/alerting/alert_rules.yaml`. Each rule specifies:

- A **condition** (event type, result, severity, count within window).
- A **severity** level (INFO, WARNING, HIGH, CRITICAL).
- **Notification channels** (Slack, PagerDuty, email).
- A **cooldown** period to prevent alert storms.
- A **group_by** field for aggregation.

Notification channel configuration (webhook URLs, API keys) is in the `alerting` section of `sentinel.yaml`.

#### Dynamic Configuration via gRPC

The correlation engine exposes a `ConfigService` gRPC endpoint that supports:

- **PushConfig:** Immediately update probe schedules, thresholds, or alerting rules for all connected agents.
- **StreamConfig:** Agents subscribe to a config stream and receive updates in real time without restart.

This enables operators to adjust sensitivity on the fly, such as increasing probe frequency for SUSPECT GPUs or adjusting EWMA parameters during a model deployment.

---

## Upgrading

### Version Compatibility

SENTINEL uses semantic versioning. During the alpha phase (0.x.y), minor version increments may include breaking changes.

| Upgrade Path | Compatibility |
|-------------|--------------|
| 0.1.0 -> 0.1.1 (patch) | Fully compatible. Rolling upgrade. |
| 0.1.x -> 0.2.0 (minor, alpha) | May include breaking config/API changes. Check release notes. |
| 0.x -> 1.0 (major) | Migration guide provided. |

### Rolling Upgrade Procedure

1. **Read the release notes** for the target version. Check for breaking changes, required migrations, and new configuration options.

2. **Upgrade data stores first** if schema migrations are required:
   ```bash
   # Run audit ledger migrations
   sentinel-audit-ledger migrate --config /etc/sentinel/config/sentinel.yaml
   ```

3. **Upgrade the correlation engine** (stateless, supports rolling restart):
   ```bash
   # Kubernetes
   kubectl set image deployment/correlation-engine \
     correlation-engine=ghcr.io/sentinel-sdc/correlation-engine:NEW_VERSION \
     -n sentinel

   # Watch rollout
   kubectl rollout status deployment/correlation-engine -n sentinel
   ```

4. **Upgrade the audit ledger writer** (brief downtime is acceptable; events are buffered in the correlation engine):
   ```bash
   kubectl set image deployment/audit-ledger-writer \
     audit-ledger=ghcr.io/sentinel-sdc/audit-ledger:NEW_VERSION \
     -n sentinel
   ```

5. **Upgrade probe agents** (rolling DaemonSet update):
   ```bash
   kubectl set image daemonset/sentinel-probe-agent \
     probe-agent=ghcr.io/sentinel-sdc/probe-agent:NEW_VERSION \
     -n sentinel
   ```

6. **Upgrade inference/training monitors** by updating the library version and restarting inference/training pods:
   ```bash
   pip install sentinel-inference==NEW_VERSION
   pip install sentinel-training==NEW_VERSION
   ```

7. **Verify** the upgrade:
   ```bash
   # Check all pods are running
   kubectl get pods -n sentinel

   # Verify probe results are flowing
   curl -s http://correlation-engine:9090/metrics | grep sentinel_probe_results_total

   # Verify audit chain integrity
   sentinel-audit-ledger verify --depth 100
   ```

### Rollback

If issues are detected after an upgrade:

```bash
# Kubernetes: rollback to previous revision
kubectl rollout undo deployment/correlation-engine -n sentinel
kubectl rollout undo deployment/audit-ledger-writer -n sentinel
kubectl rollout undo daemonset/sentinel-probe-agent -n sentinel
```

For database schema rollbacks, check the migration files in `audit-ledger/src/storage/migrations/` for down-migration instructions.
