"""
SENTINEL SDC Injector — Test Scenario Definitions

Each scenario encapsulates a specific type of silent data corruption to inject,
the expected detection layer, and a validation function to confirm that
SENTINEL detected the fault within the expected time window.

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


class InjectionType(str, enum.Enum):
    """Supported SDC injection primitives."""
    BITFLIP = "bitflip"
    STUCK_AT = "stuck_at"
    NOISE = "noise"
    MEMORY_STUCK_BIT = "memory_stuck_bit"
    TENSOR_CORE = "tensor_core"
    REGISTER = "register"
    SHARED_MEMORY = "shared_memory"


class DetectionLayer(str, enum.Enum):
    """SENTINEL detection layers that should catch each fault."""
    PROBE_AGENT = "probe_agent"
    INFERENCE_MONITOR = "inference_monitor"
    TRAINING_MONITOR = "training_monitor"
    CORRELATION_ENGINE = "correlation_engine"


@dataclass
class InjectionTarget:
    """Where to inject the fault."""
    gpu_index: int = 0
    memory_region: str = "weights"  # weights | activations | gradients
    tensor_name: Optional[str] = None
    byte_offset: int = 0
    sm_id: Optional[int] = None


@dataclass
class InjectionParameters:
    """Fault-specific parameters."""
    bit_position: int = 0
    stuck_value: float = 0.0
    sigma: float = 0.01
    xor_mask: int = 0x00000100
    smem_word_index: int = 0
    corrupt_value: float = 0.0
    duration_seconds: float = 0.0
    ramp_steps: int = 1
    count: int = 1024


@dataclass
class Scenario:
    """A complete SDC injection test scenario."""
    name: str
    description: str
    injection_type: InjectionType
    target: InjectionTarget
    parameters: InjectionParameters
    expected_detection_layer: DetectionLayer
    expected_detection_time_s: float
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["injection_type"] = self.injection_type.value
        d["expected_detection_layer"] = self.expected_detection_layer.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DetectionEvent:
    """A detection event reported by SENTINEL."""
    timestamp: float
    layer: DetectionLayer
    gpu_index: int
    confidence: float
    description: str
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating whether SENTINEL detected an injected fault."""
    scenario_name: str
    detected: bool
    detection_time_s: Optional[float]
    expected_time_s: float
    detection_layer: Optional[DetectionLayer]
    expected_layer: DetectionLayer
    layer_match: bool
    within_time_budget: bool
    details: str

    @property
    def passed(self) -> bool:
        return self.detected and self.layer_match and self.within_time_budget


def _validate_detection(
    scenario: Scenario,
    events: list[DetectionEvent],
    injection_time: float,
) -> ValidationResult:
    """
    Core validation logic: check that at least one detection event matches
    the expected layer and arrived within the expected time window.
    """
    matching = [
        e for e in events
        if e.layer == scenario.expected_detection_layer
        and e.gpu_index == scenario.target.gpu_index
    ]

    if not matching:
        return ValidationResult(
            scenario_name=scenario.name,
            detected=len(events) > 0,
            detection_time_s=None,
            expected_time_s=scenario.expected_detection_time_s,
            detection_layer=events[0].layer if events else None,
            expected_layer=scenario.expected_detection_layer,
            layer_match=False,
            within_time_budget=False,
            details=(
                f"No detection events from expected layer "
                f"'{scenario.expected_detection_layer.value}'. "
                f"Total events received: {len(events)}."
            ),
        )

    earliest = min(matching, key=lambda e: e.timestamp)
    dt = earliest.timestamp - injection_time

    return ValidationResult(
        scenario_name=scenario.name,
        detected=True,
        detection_time_s=dt,
        expected_time_s=scenario.expected_detection_time_s,
        detection_layer=earliest.layer,
        expected_layer=scenario.expected_detection_layer,
        layer_match=True,
        within_time_budget=dt <= scenario.expected_detection_time_s,
        details=(
            f"Detected by {earliest.layer.value} in {dt:.3f}s "
            f"(budget: {scenario.expected_detection_time_s:.1f}s). "
            f"Confidence: {earliest.confidence:.4f}."
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
#  Predefined Scenarios
# ────────────────────────────────────────────────────────────────────────────

SINGLE_WEIGHT_BITFLIP = Scenario(
    name="single_weight_bitflip",
    description=(
        "Flip a single bit in a random weight tensor. This is the most common "
        "form of SDC and should be caught by the probe agent within one probe "
        "cycle."
    ),
    injection_type=InjectionType.BITFLIP,
    target=InjectionTarget(
        gpu_index=0,
        memory_region="weights",
        tensor_name="model.layers.12.self_attn.q_proj.weight",
        byte_offset=8192,
    ),
    parameters=InjectionParameters(bit_position=5, count=1),
    expected_detection_layer=DetectionLayer.PROBE_AGENT,
    expected_detection_time_s=30.0,
    tags=["basic", "probe", "weight"],
)

FMA_STUCK_AT = Scenario(
    name="fma_stuck_at",
    description=(
        "Replace FMA results with a constant value on a single Streaming "
        "Multiprocessor. Simulates a hardware defect in one SM's FP unit."
    ),
    injection_type=InjectionType.STUCK_AT,
    target=InjectionTarget(gpu_index=0, sm_id=7),
    parameters=InjectionParameters(stuck_value=42.0, count=4096),
    expected_detection_layer=DetectionLayer.PROBE_AGENT,
    expected_detection_time_s=15.0,
    tags=["hardware", "probe", "sm"],
)

GRADUAL_DEGRADATION = Scenario(
    name="gradual_degradation",
    description=(
        "Inject increasing Gaussian noise over time on a single GPU to "
        "simulate gradual hardware degradation. The noise sigma ramps from "
        "1e-6 to 1e-2 over 300 seconds."
    ),
    injection_type=InjectionType.NOISE,
    target=InjectionTarget(gpu_index=2, memory_region="activations"),
    parameters=InjectionParameters(
        sigma=0.01,
        duration_seconds=300.0,
        ramp_steps=50,
        count=65536,
    ),
    expected_detection_layer=DetectionLayer.INFERENCE_MONITOR,
    expected_detection_time_s=120.0,
    tags=["gradual", "monitor", "noise"],
)

CORRELATED_FAILURE = Scenario(
    name="correlated_failure",
    description=(
        "Inject faults on multiple GPUs simultaneously to simulate a power "
        "rail or cooling failure affecting a group of devices."
    ),
    injection_type=InjectionType.NOISE,
    target=InjectionTarget(gpu_index=0, memory_region="activations"),
    parameters=InjectionParameters(sigma=0.1, count=32768),
    expected_detection_layer=DetectionLayer.CORRELATION_ENGINE,
    expected_detection_time_s=60.0,
    tags=["correlated", "multi-gpu", "correlation"],
)

BYZANTINE_FAULT = Scenario(
    name="byzantine_fault",
    description=(
        "Inject a subtle register corruption that produces plausible but "
        "incorrect results. The XOR mask targets the least significant "
        "exponent bit, causing small multiplicative errors."
    ),
    injection_type=InjectionType.REGISTER,
    target=InjectionTarget(gpu_index=0, memory_region="activations"),
    parameters=InjectionParameters(xor_mask=0x00800000, count=8192),
    expected_detection_layer=DetectionLayer.INFERENCE_MONITOR,
    expected_detection_time_s=45.0,
    tags=["byzantine", "subtle", "register"],
)

TENSOR_CORE_CORRUPTION = Scenario(
    name="tensor_core_corruption",
    description=(
        "Corrupt the output of HMMA (half-precision matrix multiply-accumulate) "
        "operations by flipping a bit in the FP16 mantissa. Targets tensor "
        "core computation specifically."
    ),
    injection_type=InjectionType.TENSOR_CORE,
    target=InjectionTarget(gpu_index=0, memory_region="activations"),
    parameters=InjectionParameters(bit_position=9, count=256 * 256),
    expected_detection_layer=DetectionLayer.PROBE_AGENT,
    expected_detection_time_s=20.0,
    tags=["tensor_core", "probe", "hmma"],
)

MEMORY_STUCK_BIT = Scenario(
    name="memory_stuck_bit",
    description=(
        "Simulate a single stuck bit in GPU global memory. The bit is forced "
        "to 1 regardless of what is written, modeling a physical HBM defect."
    ),
    injection_type=InjectionType.MEMORY_STUCK_BIT,
    target=InjectionTarget(
        gpu_index=0,
        memory_region="weights",
        byte_offset=16384,
    ),
    parameters=InjectionParameters(bit_position=3, stuck_value=1.0, count=1),
    expected_detection_layer=DetectionLayer.PROBE_AGENT,
    expected_detection_time_s=30.0,
    tags=["memory", "probe", "stuck_bit"],
)

# Ordered list of all predefined scenarios.
ALL_SCENARIOS: list[Scenario] = [
    SINGLE_WEIGHT_BITFLIP,
    FMA_STUCK_AT,
    GRADUAL_DEGRADATION,
    CORRELATED_FAILURE,
    BYZANTINE_FAULT,
    TENSOR_CORE_CORRUPTION,
    MEMORY_STUCK_BIT,
]


def get_scenario(name: str) -> Scenario:
    """Look up a predefined scenario by name."""
    for s in ALL_SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"Unknown scenario: {name!r}. "
                   f"Available: {[s.name for s in ALL_SCENARIOS]}")


def validate_scenario(
    scenario: Scenario,
    events: list[DetectionEvent],
    injection_time: float,
) -> ValidationResult:
    """Validate whether SENTINEL correctly detected an injected fault."""
    return _validate_detection(scenario, events, injection_time)


if __name__ == "__main__":
    print("SENTINEL SDC Injection Scenarios")
    print("=" * 60)
    for s in ALL_SCENARIOS:
        print(f"\n  {s.name}")
        print(f"    Type:      {s.injection_type.value}")
        print(f"    Layer:     {s.expected_detection_layer.value}")
        print(f"    Budget:    {s.expected_detection_time_s}s")
        print(f"    Tags:      {', '.join(s.tags)}")
        print(f"    {s.description[:80]}...")
