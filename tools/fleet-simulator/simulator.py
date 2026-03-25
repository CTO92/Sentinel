"""
SENTINEL Fleet Simulator

Simulates large GPU fleets to test SENTINEL at scale. Generates synthetic
probe results, telemetry, and anomaly events that are streamed to the
Correlation Engine via gRPC.

Features:
  - Configurable fleet size (up to 10,000+ GPUs)
  - Realistic telemetry: temperature, power, utilization with temporal patterns
  - Health state machine: healthy -> degrading -> faulty
  - YAML scenario files for reproducible test configurations
  - gRPC streaming to the Correlation Engine

Usage:
    python simulator.py --scenario scenarios/healthy_fleet.yaml
    python simulator.py --scenario scenarios/single_gpu_degradation.yaml --grpc localhost:50051
    python simulator.py --fleet-size 5000 --duration 3600 --sdc-rate 0.001

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sentinel.fleet-simulator")


# ────────────────────────────────────────────────────────────────────────────
#  GPU Health Model
# ────────────────────────────────────────────────────────────────────────────

class GpuHealth(str, enum.Enum):
    HEALTHY = "healthy"
    DEGRADING = "degrading"
    FAULTY = "faulty"


@dataclass
class TelemetrySnapshot:
    """A single telemetry reading for a GPU."""
    timestamp: float
    gpu_id: str
    node_id: str
    temperature_c: float
    power_watts: float
    utilization_pct: float
    memory_used_gb: float
    memory_total_gb: float
    clock_mhz: int
    health: GpuHealth
    ecc_sbe_count: int = 0
    ecc_dbe_count: int = 0
    pcie_replay_count: int = 0


@dataclass
class ProbeResult:
    """A synthetic probe result from a simulated GPU."""
    timestamp: float
    gpu_id: str
    node_id: str
    probe_type: str
    passed: bool
    deviation: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    """An anomaly event to send to the Correlation Engine."""
    timestamp: float
    event_id: str
    gpu_id: str
    node_id: str
    severity: str  # info, warning, critical
    event_type: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)


class SimulatedGpu:
    """
    Simulates a single GPU with a health state machine and realistic
    telemetry generation.
    """

    def __init__(
        self,
        gpu_id: str,
        node_id: str,
        gpu_index: int = 0,
        initial_health: GpuHealth = GpuHealth.HEALTHY,
        sdc_rate: float = 0.0,
        degradation_start_s: Optional[float] = None,
        degradation_duration_s: float = 86400.0,
        memory_total_gb: float = 80.0,
        base_temperature: float = 45.0,
        base_power: float = 250.0,
        base_clock: int = 1410,
    ):
        self.gpu_id = gpu_id
        self.node_id = node_id
        self.gpu_index = gpu_index
        self.health = initial_health
        self.sdc_rate = sdc_rate
        self.degradation_start_s = degradation_start_s
        self.degradation_duration_s = degradation_duration_s
        self.memory_total_gb = memory_total_gb
        self.base_temperature = base_temperature
        self.base_power = base_power
        self.base_clock = base_clock

        self.ecc_sbe_count = 0
        self.ecc_dbe_count = 0
        self.pcie_replay_count = 0
        self._rng = random.Random(hash(gpu_id))
        self._creation_time = time.time()

    def _time_of_day_factor(self, t: float) -> float:
        """Simulate diurnal utilization pattern (higher during business hours)."""
        hour = (t / 3600) % 24
        # Peak at 14:00, trough at 04:00
        return 0.5 + 0.4 * math.sin(math.pi * (hour - 4) / 12)

    def _thermal_cycling(self, t: float) -> float:
        """Simulate thermal oscillation (HVAC cycling)."""
        period = 1800  # 30 minutes
        return 2.0 * math.sin(2 * math.pi * t / period)

    def _update_health(self, sim_time: float) -> None:
        """Transition health state based on simulation time."""
        if self.degradation_start_s is None:
            return
        if sim_time < self.degradation_start_s:
            return

        elapsed = sim_time - self.degradation_start_s
        progress = min(1.0, elapsed / self.degradation_duration_s)

        if progress < 0.5:
            self.health = GpuHealth.DEGRADING
        else:
            self.health = GpuHealth.FAULTY

    def _current_sdc_rate(self, sim_time: float) -> float:
        """SDC rate increases as GPU degrades."""
        if self.health == GpuHealth.HEALTHY:
            return self.sdc_rate
        elif self.health == GpuHealth.DEGRADING:
            if self.degradation_start_s is not None:
                elapsed = sim_time - self.degradation_start_s
                progress = min(1.0, elapsed / self.degradation_duration_s)
                return self.sdc_rate + progress * 0.1
            return self.sdc_rate * 10
        else:  # FAULTY
            return min(1.0, self.sdc_rate * 100)

    def generate_telemetry(self, sim_time: float) -> TelemetrySnapshot:
        """Generate a telemetry snapshot for the current simulation time."""
        self._update_health(sim_time)

        utilization = self._time_of_day_factor(sim_time)
        utilization = max(0.0, min(1.0, utilization + self._rng.gauss(0, 0.05)))

        temp = (self.base_temperature
                + utilization * 30.0
                + self._thermal_cycling(sim_time)
                + self._rng.gauss(0, 1.0))

        if self.health == GpuHealth.FAULTY:
            temp += self._rng.uniform(5, 15)

        power = (self.base_power * (0.3 + 0.7 * utilization)
                 + self._rng.gauss(0, 5.0))

        mem_used = self.memory_total_gb * (0.2 + 0.6 * utilization
                                            + self._rng.gauss(0, 0.02))
        mem_used = max(0, min(self.memory_total_gb, mem_used))

        clock = self.base_clock
        if self.health == GpuHealth.FAULTY:
            clock = int(clock * 0.85)

        # Stochastic ECC errors
        if self.health != GpuHealth.HEALTHY and self._rng.random() < 0.01:
            self.ecc_sbe_count += 1
        if self.health == GpuHealth.FAULTY and self._rng.random() < 0.001:
            self.ecc_dbe_count += 1

        return TelemetrySnapshot(
            timestamp=time.time(),
            gpu_id=self.gpu_id,
            node_id=self.node_id,
            temperature_c=round(temp, 1),
            power_watts=round(power, 1),
            utilization_pct=round(utilization * 100, 1),
            memory_used_gb=round(mem_used, 2),
            memory_total_gb=self.memory_total_gb,
            clock_mhz=clock,
            health=self.health,
            ecc_sbe_count=self.ecc_sbe_count,
            ecc_dbe_count=self.ecc_dbe_count,
            pcie_replay_count=self.pcie_replay_count,
        )

    def generate_probe_result(self, sim_time: float) -> ProbeResult:
        """Generate a synthetic probe result."""
        self._update_health(sim_time)
        sdc_rate = self._current_sdc_rate(sim_time)

        # Determine if this probe encounters SDC
        has_sdc = self._rng.random() < sdc_rate

        if has_sdc:
            probe_type = self._rng.choice(["fma", "tensor_core", "memory", "transcendental"])
            deviation = self._rng.uniform(0.001, 1.0)
            if self.health == GpuHealth.FAULTY:
                deviation = self._rng.uniform(0.1, 10.0)
            passed = False
        else:
            probe_type = self._rng.choice(["fma", "tensor_core", "memory", "transcendental"])
            deviation = self._rng.uniform(0, 1e-7)
            passed = True

        return ProbeResult(
            timestamp=time.time(),
            gpu_id=self.gpu_id,
            node_id=self.node_id,
            probe_type=probe_type,
            passed=passed,
            deviation=deviation,
            details={
                "health": self.health.value,
                "sdc_rate": sdc_rate,
            },
        )


# ────────────────────────────────────────────────────────────────────────────
#  Fleet Simulator
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FleetConfig:
    """Configuration for a fleet simulation."""
    fleet_size: int = 100
    duration_seconds: float = 3600.0
    telemetry_interval_s: float = 10.0
    probe_interval_s: float = 30.0
    sdc_rate: float = 0.0
    gpus_per_node: int = 8
    degrading_gpus: list[dict[str, Any]] = field(default_factory=list)
    correlated_groups: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> FleetConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = data.get("simulation", data)
        return cls(
            fleet_size=config.get("fleet_size", 100),
            duration_seconds=config.get("duration_seconds", 3600),
            telemetry_interval_s=config.get("telemetry_interval_s", 10),
            probe_interval_s=config.get("probe_interval_s", 30),
            sdc_rate=config.get("sdc_rate", 0.0),
            gpus_per_node=config.get("gpus_per_node", 8),
            degrading_gpus=config.get("degrading_gpus", []),
            correlated_groups=config.get("correlated_groups", []),
            seed=config.get("seed", 42),
        )


class FleetSimulator:
    """
    Simulates a fleet of GPUs and generates synthetic telemetry, probe
    results, and anomaly events.
    """

    def __init__(self, config: FleetConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        self.gpus: list[SimulatedGpu] = []
        self._build_fleet()

    def _build_fleet(self) -> None:
        """Create SimulatedGpu instances based on configuration."""
        degrading_indices: dict[int, dict[str, Any]] = {}
        for dg in self.config.degrading_gpus:
            degrading_indices[dg["gpu_index"]] = dg

        # Correlated groups: assign shared parameters
        correlated_indices: dict[int, dict[str, Any]] = {}
        for group in self.config.correlated_groups:
            for idx in group.get("gpu_indices", []):
                correlated_indices[idx] = group

        for i in range(self.config.fleet_size):
            node_idx = i // self.config.gpus_per_node
            gpu_in_node = i % self.config.gpus_per_node
            node_id = f"node-{node_idx:04d}"
            gpu_id = f"gpu-{node_id}-{gpu_in_node}"

            kwargs: dict[str, Any] = {
                "gpu_id": gpu_id,
                "node_id": node_id,
                "gpu_index": i,
                "sdc_rate": self.config.sdc_rate,
            }

            if i in degrading_indices:
                dg = degrading_indices[i]
                kwargs["degradation_start_s"] = dg.get("degradation_start_s", 0)
                kwargs["degradation_duration_s"] = dg.get(
                    "degradation_duration_s", 86400
                )
                kwargs["sdc_rate"] = dg.get("sdc_rate", 0.001)

            if i in correlated_indices:
                cg = correlated_indices[i]
                kwargs["degradation_start_s"] = cg.get("failure_start_s", 0)
                kwargs["degradation_duration_s"] = cg.get(
                    "failure_duration_s", 3600
                )
                kwargs["sdc_rate"] = cg.get("sdc_rate", 0.01)

            self.gpus.append(SimulatedGpu(**kwargs))

        log.info(
            "Fleet initialized: %d GPUs across %d nodes",
            len(self.gpus),
            (self.config.fleet_size + self.config.gpus_per_node - 1)
            // self.config.gpus_per_node,
        )

    def generate_events(
        self,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generator that yields telemetry and probe events over the
        configured simulation duration.
        """
        start_time = time.time()
        sim_time = 0.0

        next_telemetry = 0.0
        next_probe = 0.0

        while sim_time < self.config.duration_seconds:
            # Telemetry tick
            if sim_time >= next_telemetry:
                for gpu in self.gpus:
                    telem = gpu.generate_telemetry(sim_time)
                    yield {
                        "type": "telemetry",
                        "sim_time": sim_time,
                        "data": {
                            "timestamp": telem.timestamp,
                            "gpu_id": telem.gpu_id,
                            "node_id": telem.node_id,
                            "temperature_c": telem.temperature_c,
                            "power_watts": telem.power_watts,
                            "utilization_pct": telem.utilization_pct,
                            "memory_used_gb": telem.memory_used_gb,
                            "memory_total_gb": telem.memory_total_gb,
                            "clock_mhz": telem.clock_mhz,
                            "health": telem.health.value,
                            "ecc_sbe_count": telem.ecc_sbe_count,
                            "ecc_dbe_count": telem.ecc_dbe_count,
                        },
                    }
                next_telemetry += self.config.telemetry_interval_s

            # Probe tick
            if sim_time >= next_probe:
                for gpu in self.gpus:
                    probe = gpu.generate_probe_result(sim_time)
                    yield {
                        "type": "probe_result",
                        "sim_time": sim_time,
                        "data": {
                            "timestamp": probe.timestamp,
                            "gpu_id": probe.gpu_id,
                            "node_id": probe.node_id,
                            "probe_type": probe.probe_type,
                            "passed": probe.passed,
                            "deviation": probe.deviation,
                        },
                    }

                    # Generate anomaly events for failed probes
                    if not probe.passed:
                        anomaly = AnomalyEvent(
                            timestamp=probe.timestamp,
                            event_id=str(uuid.uuid4()),
                            gpu_id=probe.gpu_id,
                            node_id=probe.node_id,
                            severity="warning" if probe.deviation < 0.5 else "critical",
                            event_type="sdc_detected",
                            description=(
                                f"Probe {probe.probe_type} failed on {probe.gpu_id} "
                                f"with deviation {probe.deviation:.6f}"
                            ),
                            evidence={
                                "probe_type": probe.probe_type,
                                "deviation": probe.deviation,
                                "health": gpu.health.value,
                            },
                        )
                        yield {
                            "type": "anomaly",
                            "sim_time": sim_time,
                            "data": {
                                "event_id": anomaly.event_id,
                                "timestamp": anomaly.timestamp,
                                "gpu_id": anomaly.gpu_id,
                                "node_id": anomaly.node_id,
                                "severity": anomaly.severity,
                                "event_type": anomaly.event_type,
                                "description": anomaly.description,
                                "evidence": anomaly.evidence,
                            },
                        }

                next_probe += self.config.probe_interval_s

            # Advance simulation time (use smaller of the two intervals)
            step = min(
                self.config.telemetry_interval_s,
                self.config.probe_interval_s,
            )
            sim_time += step

    def stream_to_grpc(self, endpoint: str) -> None:
        """Stream events to the Correlation Engine via gRPC."""
        try:
            import grpc
            from google.protobuf.timestamp_pb2 import Timestamp
        except ImportError:
            log.error(
                "gRPC dependencies not installed. "
                "Install with: pip install grpcio grpcio-tools protobuf"
            )
            return

        log.info("Connecting to Correlation Engine at %s...", endpoint)

        # Import generated protobuf stubs
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "proto" / "gen" / "python"))
            from sentinel.v1 import correlation_pb2, correlation_pb2_grpc
        except ImportError:
            log.error(
                "Proto stubs not found. Generate with: "
                "cd proto && buf generate"
            )
            # Fall back to JSON streaming
            log.info("Falling back to JSON line streaming to stdout.")
            self._stream_json()
            return

        channel = grpc.insecure_channel(endpoint)
        stub = correlation_pb2_grpc.CorrelationServiceStub(channel)

        event_count = 0
        anomaly_count = 0

        for event in self.generate_events():
            if event["type"] == "anomaly":
                try:
                    ts = Timestamp()
                    ts.FromMilliseconds(int(event["data"]["timestamp"] * 1000))

                    request = correlation_pb2.ReportAnomalyRequest(
                        event_id=event["data"]["event_id"],
                        gpu_id=event["data"]["gpu_id"],
                        node_id=event["data"]["node_id"],
                        severity=event["data"]["severity"],
                        event_type=event["data"]["event_type"],
                        description=event["data"]["description"],
                        timestamp=ts,
                    )
                    stub.ReportAnomaly(request)
                    anomaly_count += 1
                except grpc.RpcError as e:
                    log.warning("gRPC error: %s", e)

            event_count += 1
            if event_count % 10000 == 0:
                log.info(
                    "Streamed %d events (%d anomalies) at sim_time=%.0fs",
                    event_count,
                    anomaly_count,
                    event["sim_time"],
                )

        log.info(
            "Simulation complete. Total events: %d, anomalies: %d",
            event_count,
            anomaly_count,
        )
        channel.close()

    def _stream_json(self) -> None:
        """Fall back: stream events as JSON lines to stdout."""
        event_count = 0
        for event in self.generate_events():
            print(json.dumps(event))
            event_count += 1
            if event_count % 10000 == 0:
                log.info("Streamed %d events", event_count)
        log.info("Simulation complete. Total events: %d", event_count)

    def run(self, grpc_endpoint: Optional[str] = None) -> None:
        """Run the simulation, optionally streaming to gRPC."""
        if grpc_endpoint:
            self.stream_to_grpc(grpc_endpoint)
        else:
            self._stream_json()


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SENTINEL Fleet Simulator"
    )
    parser.add_argument(
        "--scenario",
        help="Path to YAML scenario file",
    )
    parser.add_argument(
        "--fleet-size",
        type=int,
        default=100,
        help="Number of GPUs to simulate (default: 100)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600,
        help="Simulation duration in seconds (default: 3600)",
    )
    parser.add_argument(
        "--sdc-rate",
        type=float,
        default=0.0,
        help="Base SDC rate per probe (default: 0.0)",
    )
    parser.add_argument(
        "--grpc",
        default=None,
        help="gRPC endpoint for Correlation Engine (e.g. localhost:50051)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.scenario:
        config = FleetConfig.from_yaml(args.scenario)
    else:
        config = FleetConfig(
            fleet_size=args.fleet_size,
            duration_seconds=args.duration,
            sdc_rate=args.sdc_rate,
            seed=args.seed,
        )

    simulator = FleetSimulator(config)
    simulator.run(grpc_endpoint=args.grpc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
