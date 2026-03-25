"""Configuration models for the SENTINEL Inference Monitor.

Uses Pydantic v2 for validation. Supports loading from YAML files,
environment variables (prefixed SENTINEL_), and gRPC dynamic updates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field, model_validator

logger = structlog.get_logger(__name__)


class EWMAConfig(BaseModel):
    """Parameters for Exponentially Weighted Moving Average control charts."""

    lambda_: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        alias="lambda",
        description="EWMA smoothing factor. Lower = more smoothing.",
    )
    L: float = Field(
        default=3.5,
        ge=1.0,
        le=6.0,
        description="Control limit width in sigma units.",
    )
    burn_in: int = Field(
        default=1000,
        ge=10,
        description="Number of samples before alerting is enabled.",
    )

    model_config = {"populate_by_name": True}


class LogitAnalyzerConfig(BaseModel):
    """Configuration for the logit statistics analyzer."""

    enabled: bool = True
    ewma: EWMAConfig = Field(default_factory=EWMAConfig)
    tracked_stats: list[str] = Field(
        default=["mean", "variance", "kurtosis", "min", "max", "entropy"],
        description="Which logit statistics to track with EWMA.",
    )


class EntropyAnalyzerConfig(BaseModel):
    """Configuration for the entropy analyzer."""

    enabled: bool = True
    ewma: EWMAConfig = Field(default_factory=EWMAConfig)
    min_entropy: float = Field(
        default=0.5,
        ge=0.0,
        description="Entropy below this threshold triggers collapse alert.",
    )
    max_entropy: float = Field(
        default=15.0,
        ge=0.0,
        description="Entropy above this threshold triggers explosion alert.",
    )


class KLDivergenceConfig(BaseModel):
    """Configuration for cross-replica KL divergence detection."""

    enabled: bool = True
    threshold_nats: float = Field(
        default=0.01,
        ge=0.0,
        description="JSD threshold in nats above which an anomaly is raised.",
    )
    match_window_seconds: float = Field(
        default=2.0,
        ge=0.1,
        description="Time window for matching requests across replicas.",
    )
    use_jsd: bool = Field(
        default=True,
        description="Use symmetric Jensen-Shannon divergence instead of raw KL.",
    )


class SpectralAnalyzerConfig(BaseModel):
    """Configuration for FFT-based spectral analysis."""

    enabled: bool = True
    high_freq_ratio_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of energy in top-quarter frequencies that triggers alert.",
    )
    baseline_window: int = Field(
        default=500,
        ge=10,
        description="Number of samples for building baseline PSD.",
    )


class StatisticalTestsConfig(BaseModel):
    """Configuration for KS and Anderson-Darling tests."""

    enabled: bool = True
    ks_p_value_threshold: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="KS test p-value below which an anomaly is raised.",
    )
    ad_significance_level: int = Field(
        default=1,
        ge=0,
        le=4,
        description="Index into Anderson-Darling significance levels (0=15%, 1=10%, 2=5%, 3=2.5%, 4=1%).",
    )
    baseline_size: int = Field(
        default=5000,
        ge=100,
        description="Number of samples kept for the rolling baseline distribution.",
    )


class FingerprintConfig(BaseModel):
    """Configuration for tensor fingerprinting."""

    projection_dim: int = Field(
        default=64,
        ge=8,
        description="Number of random projection dimensions.",
    )
    bits_per_dim: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Quantization bits per projected dimension.",
    )
    seed: int = Field(
        default=42,
        description="Fixed random seed for reproducible projections.",
    )


class SketchConfig(BaseModel):
    """Configuration for Count-Min Sketch and HyperLogLog."""

    cms_width: int = Field(default=2048, ge=64)
    cms_depth: int = Field(default=5, ge=2)
    hll_precision: int = Field(default=14, ge=4, le=18)
    shift_threshold: float = Field(
        default=0.05,
        ge=0.0,
        description="Relative frequency shift that triggers alert.",
    )


class SamplingConfig(BaseModel):
    """Tensor sampling strategy configuration."""

    rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of inferences to sample (0.01 = 1%).",
    )
    max_queue_size: int = Field(
        default=1000,
        ge=10,
        description="Maximum pending samples in the async queue.",
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of analyzer worker tasks.",
    )


class GRPCConfig(BaseModel):
    """gRPC client configuration for Correlation Engine reporting."""

    endpoint: str = Field(
        default="localhost:50051",
        description="Correlation Engine gRPC endpoint.",
    )
    batch_size: int = Field(default=100, ge=1)
    flush_interval_ms: int = Field(default=500, ge=50)
    max_retries: int = Field(default=5, ge=0)
    initial_backoff_ms: int = Field(default=100, ge=10)
    max_backoff_ms: int = Field(default=30000, ge=100)
    enable_mtls: bool = False
    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None


class InterceptorConfig(BaseModel):
    """Configuration for the tensor interceptor layer."""

    type: Literal["triton", "vllm", "trtllm", "generic"] = "generic"
    shm_name: str = Field(
        default="/sentinel_tensor_shm",
        description="Name of shared memory region (for generic interceptor).",
    )
    shm_size_bytes: int = Field(
        default=64 * 1024 * 1024,
        description="Size of shared memory region in bytes.",
    )
    triton_model_repository: str | None = None
    vllm_hook_module: str | None = None


class HealthCheckConfig(BaseModel):
    """K8s health check server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)


class MetricsConfig(BaseModel):
    """Prometheus metrics server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=9090, ge=1, le=65535)


class MonitorConfig(BaseModel):
    """Top-level configuration for the SENTINEL Inference Monitor."""

    node_id: str = Field(default="", description="Unique identifier for this node.")
    gpu_id: int = Field(default=0, ge=0, description="GPU index being monitored.")
    replica_id: str = Field(default="", description="Replica identifier for cross-replica checks.")

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    interceptor: InterceptorConfig = Field(default_factory=InterceptorConfig)
    logit_analyzer: LogitAnalyzerConfig = Field(default_factory=LogitAnalyzerConfig)
    entropy_analyzer: EntropyAnalyzerConfig = Field(default_factory=EntropyAnalyzerConfig)
    kl_divergence: KLDivergenceConfig = Field(default_factory=KLDivergenceConfig)
    spectral_analyzer: SpectralAnalyzerConfig = Field(default_factory=SpectralAnalyzerConfig)
    statistical_tests: StatisticalTestsConfig = Field(default_factory=StatisticalTestsConfig)
    fingerprint: FingerprintConfig = Field(default_factory=FingerprintConfig)
    sketch: SketchConfig = Field(default_factory=SketchConfig)
    grpc: GRPCConfig = Field(default_factory=GRPCConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @model_validator(mode="after")
    def _set_defaults(self) -> "MonitorConfig":
        if not self.node_id:
            self.node_id = os.environ.get("HOSTNAME", "unknown")
        if not self.replica_id:
            self.replica_id = os.environ.get("POD_NAME", self.node_id)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MonitorConfig":
        """Load configuration from a YAML file, with env-var overrides."""
        import yaml  # type: ignore[import-untyped]

        p = Path(path)
        if not p.exists():
            logger.warning("config_file_not_found", path=str(p))
            return cls()
        with open(p) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        """Build config from environment variables prefixed with SENTINEL_."""
        overrides: dict[str, str] = {}
        for key, val in os.environ.items():
            if key.startswith("SENTINEL_"):
                overrides[key.removeprefix("SENTINEL_").lower()] = val
        # Flatten simple top-level overrides
        cfg_dict: dict[str, object] = {}
        if "sample_rate" in overrides:
            cfg_dict.setdefault("sampling", {})
            cfg_dict["sampling"]["rate"] = float(overrides["sample_rate"])  # type: ignore[index]
        if "grpc_endpoint" in overrides:
            cfg_dict.setdefault("grpc", {})
            cfg_dict["grpc"]["endpoint"] = overrides["grpc_endpoint"]  # type: ignore[index]
        if "node_id" in overrides:
            cfg_dict["node_id"] = overrides["node_id"]
        if "gpu_id" in overrides:
            cfg_dict["gpu_id"] = int(overrides["gpu_id"])
        return cls.model_validate(cfg_dict)
