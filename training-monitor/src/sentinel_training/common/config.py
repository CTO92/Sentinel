"""Pydantic v2 configuration for Sentinel Training Monitor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, model_validator

logger = structlog.get_logger(__name__)


class GradientNormConfig(BaseModel):
    """Configuration for gradient norm monitoring."""

    enabled: bool = True
    ewma_lambda: float = Field(default=0.05, gt=0.0, le=1.0)
    sigma_multiplier: float = Field(default=4.0, gt=0.0)
    warmup_steps: int = Field(default=50, ge=1)
    per_layer: bool = True
    track_param_groups: bool = True
    mixed_precision_aware: bool = True


class LossTrackingConfig(BaseModel):
    """Configuration for loss trajectory monitoring."""

    enabled: bool = True
    ar_order: int = Field(default=5, ge=1, le=20)
    sigma_multiplier: float = Field(default=3.0, gt=0.0)
    warmup_steps: int = Field(default=20, ge=1)
    nan_detection: bool = True
    multi_task: bool = False
    plateau_window: int = Field(default=50, ge=5)
    plateau_tolerance: float = Field(default=1e-7, gt=0.0)


class CrossRankDivergenceConfig(BaseModel):
    """Configuration for cross-rank divergence detection."""

    enabled: bool = True
    check_interval: int = Field(default=100, ge=1)
    hash_seed: int = 42
    projection_dim: int = Field(default=256, ge=16)
    tolerance: float = Field(default=1e-6, gt=0.0)
    supports_fsdp: bool = True
    supports_deepspeed: bool = True


class CheckpointValidationConfig(BaseModel):
    """Configuration for checkpoint validation."""

    enabled: bool = True
    projection_seed: int = 12345
    projection_dim: int = Field(default=512, ge=32)
    max_param_divergence: float = Field(default=0.1, gt=0.0)
    sha256_verify: bool = True


class RecomputationConfig(BaseModel):
    """Configuration for SDC recomputation verification."""

    enabled: bool = True
    replay_steps: int = Field(default=10, ge=1)
    auto_rollback: bool = False
    divergence_threshold: float = Field(default=1e-5, gt=0.0)
    target_device: str | None = None
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class GrpcConfig(BaseModel):
    """Configuration for gRPC client to Correlation Engine."""

    endpoint: str = "localhost:50051"
    batch_size: int = Field(default=32, ge=1)
    flush_interval_seconds: float = Field(default=5.0, gt=0.0)
    max_backoff_seconds: float = Field(default=60.0, gt=0.0)
    mtls_enabled: bool = False
    cert_path: str | None = None
    key_path: str | None = None
    ca_path: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0.0)


class MetricsConfig(BaseModel):
    """Configuration for Prometheus metrics."""

    enabled: bool = True
    port: int = Field(default=9090, ge=1, le=65535)
    prefix: str = "sentinel_training"


class SentinelConfig(BaseModel):
    """Root configuration for Sentinel Training Monitor."""

    gradient_norm: GradientNormConfig = Field(default_factory=GradientNormConfig)
    loss_tracking: LossTrackingConfig = Field(default_factory=LossTrackingConfig)
    cross_rank_divergence: CrossRankDivergenceConfig = Field(
        default_factory=CrossRankDivergenceConfig
    )
    checkpoint_validation: CheckpointValidationConfig = Field(
        default_factory=CheckpointValidationConfig
    )
    recomputation: RecomputationConfig = Field(default_factory=RecomputationConfig)
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    log_level: str = "INFO"
    node_id: str | None = None
    cluster_id: str | None = None

    @model_validator(mode="after")
    def _resolve_env_overrides(self) -> SentinelConfig:
        """Apply environment variable overrides with SENTINEL_ prefix."""
        env_map: dict[str, tuple[str, str, type[Any]]] = {
            "SENTINEL_GRPC_ENDPOINT": ("grpc", "endpoint", str),
            "SENTINEL_GRPC_MTLS_ENABLED": ("grpc", "mtls_enabled", bool),
            "SENTINEL_GRPC_CERT_PATH": ("grpc", "cert_path", str),
            "SENTINEL_GRPC_KEY_PATH": ("grpc", "key_path", str),
            "SENTINEL_GRPC_CA_PATH": ("grpc", "ca_path", str),
            "SENTINEL_LOG_LEVEL": ("", "log_level", str),
            "SENTINEL_NODE_ID": ("", "node_id", str),
            "SENTINEL_CLUSTER_ID": ("", "cluster_id", str),
            "SENTINEL_GRADIENT_EWMA_LAMBDA": ("gradient_norm", "ewma_lambda", float),
            "SENTINEL_GRADIENT_SIGMA": ("gradient_norm", "sigma_multiplier", float),
            "SENTINEL_LOSS_AR_ORDER": ("loss_tracking", "ar_order", int),
            "SENTINEL_LOSS_SIGMA": ("loss_tracking", "sigma_multiplier", float),
            "SENTINEL_CROSS_RANK_INTERVAL": ("cross_rank_divergence", "check_interval", int),
            "SENTINEL_RECOMPUTE_STEPS": ("recomputation", "replay_steps", int),
            "SENTINEL_RECOMPUTE_AUTO_ROLLBACK": ("recomputation", "auto_rollback", bool),
            "SENTINEL_METRICS_PORT": ("metrics", "port", int),
        }
        for env_key, (section, field, typ) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                converted: Any = val.lower() in ("true", "1", "yes") if typ is bool else typ(val)
                if section:
                    setattr(getattr(self, section), field, converted)
                else:
                    setattr(self, field, converted)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SentinelConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed SentinelConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ImportError: If PyYAML is not installed.
        """
        import yaml  # type: ignore[import-untyped]

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        logger.info("loaded_config", path=str(path))
        return cls.model_validate(data)

    @classmethod
    def from_env(cls) -> SentinelConfig:
        """Create config from environment variables only."""
        return cls()
