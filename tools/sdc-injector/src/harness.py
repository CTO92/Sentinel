"""
SENTINEL SDC Injector — Test Harness

Orchestrates injection scenarios, monitors the SENTINEL detection pipeline,
and produces structured test reports.

Usage:
    python harness.py --sentinel-api http://localhost:8080 --scenarios all
    python harness.py --sentinel-api http://localhost:8080 --scenarios single_weight_bitflip,fma_stuck_at
    python harness.py --sentinel-api http://localhost:8080 --scenario-file custom.json --report report.json

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from scenarios import (
    ALL_SCENARIOS,
    DetectionEvent,
    DetectionLayer,
    InjectionType,
    Scenario,
    ValidationResult,
    get_scenario,
    validate_scenario,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("sentinel.harness")


# ────────────────────────────────────────────────────────────────────────────
#  Data classes
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Complete result for one scenario execution."""
    scenario_name: str
    injection_type: str
    started_at: float
    finished_at: float
    injection_succeeded: bool
    validation: Optional[ValidationResult]
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return (
            self.injection_succeeded
            and self.validation is not None
            and self.validation.passed
        )

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        d = {
            "scenario_name": self.scenario_name,
            "injection_type": self.injection_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "injection_succeeded": self.injection_succeeded,
            "passed": self.passed,
            "error": self.error,
        }
        if self.validation:
            d["validation"] = asdict(self.validation)
            # Convert enum values for JSON serialization
            if self.validation.detection_layer:
                d["validation"]["detection_layer"] = self.validation.detection_layer.value
            d["validation"]["expected_layer"] = self.validation.expected_layer.value
        return d


@dataclass
class TestReport:
    """Aggregate report for all executed scenarios."""
    started_at: float
    finished_at: float
    sentinel_api: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "duration_s": round(self.finished_at - self.started_at, 3),
            },
            "sentinel_api": self.sentinel_api,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def print_summary(self) -> None:
        total_time = self.finished_at - self.started_at
        print("\n" + "=" * 70)
        print("SENTINEL SDC Injection Test Report")
        print("=" * 70)
        print(f"  Total scenarios:  {self.total}")
        print(f"  Passed:           {self.passed}")
        print(f"  Failed:           {self.failed}")
        print(f"  Duration:         {total_time:.1f}s")
        print("-" * 70)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            det_time = ""
            if r.validation and r.validation.detection_time_s is not None:
                det_time = f" ({r.validation.detection_time_s:.3f}s)"
            err = f" [{r.error}]" if r.error else ""
            print(f"  [{status}] {r.scenario_name}{det_time}{err}")

        print("=" * 70)


# ────────────────────────────────────────────────────────────────────────────
#  SENTINEL API client
# ────────────────────────────────────────────────────────────────────────────

class SentinelAPIClient:
    """Thin wrapper around the SENTINEL REST API for detection event polling."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def health_check(self) -> bool:
        try:
            r = self.session.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    def get_detection_events(
        self,
        since_timestamp: float,
        gpu_index: Optional[int] = None,
    ) -> list[DetectionEvent]:
        params: dict[str, Any] = {"since": since_timestamp}
        if gpu_index is not None:
            params["gpu_index"] = gpu_index

        try:
            r = self.session.get(
                f"{self.base_url}/api/v1/detections",
                params=params,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Failed to fetch detection events: %s", exc)
            return []

        events: list[DetectionEvent] = []
        for item in r.json().get("events", []):
            try:
                events.append(
                    DetectionEvent(
                        timestamp=float(item["timestamp"]),
                        layer=DetectionLayer(item["layer"]),
                        gpu_index=int(item["gpu_index"]),
                        confidence=float(item.get("confidence", 0.0)),
                        description=item.get("description", ""),
                        raw_data=item,
                    )
                )
            except (KeyError, ValueError) as exc:
                log.debug("Skipping malformed event: %s", exc)
        return events

    def clear_events(self) -> None:
        try:
            self.session.delete(
                f"{self.base_url}/api/v1/detections", timeout=self.timeout
            )
        except requests.RequestException as exc:
            log.warning("Failed to clear events: %s", exc)


# ────────────────────────────────────────────────────────────────────────────
#  Injector interface (loads the native CUDA library)
# ────────────────────────────────────────────────────────────────────────────

class NativeInjector:
    """Load and call the sdc_injector shared library via ctypes."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            search_paths = [
                Path(__file__).parent.parent / "build" / "libsdc_injector.so",
                Path("/usr/local/lib/libsdc_injector.so"),
                Path("libsdc_injector.so"),
            ]
            for p in search_paths:
                if p.exists():
                    lib_path = str(p)
                    break
            if lib_path is None:
                raise FileNotFoundError(
                    "Cannot find libsdc_injector.so. Build with CMake first."
                )
        self.lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()

    def _setup_prototypes(self) -> None:
        self.lib.sdc_injector_enable.argtypes = [ctypes.c_bool]
        self.lib.sdc_injector_enable.restype = None

        self.lib.inject_bitflip.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self.lib.inject_bitflip.restype = ctypes.c_int

        self.lib.inject_stuck_at.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
        self.lib.inject_stuck_at.restype = ctypes.c_int

        self.lib.inject_noise.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        ]
        self.lib.inject_noise.restype = ctypes.c_int

        self.lib.inject_tensor_core_corruption.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        self.lib.inject_tensor_core_corruption.restype = ctypes.c_int

        self.lib.inject_memory_stuck_bit.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int
        ]
        self.lib.inject_memory_stuck_bit.restype = ctypes.c_int

        self.lib.inject_register_corruption.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
        ]
        self.lib.inject_register_corruption.restype = ctypes.c_int

        self.lib.inject_shared_memory_corruption.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
        self.lib.inject_shared_memory_corruption.restype = ctypes.c_int

    def enable(self) -> None:
        self.lib.sdc_injector_enable(True)

    def disable(self) -> None:
        self.lib.sdc_injector_enable(False)


# ────────────────────────────────────────────────────────────────────────────
#  Test Harness
# ────────────────────────────────────────────────────────────────────────────

class TestHarness:
    """
    Orchestrates SDC injection test scenarios.

    Workflow for each scenario:
    1. Clear SENTINEL detection events.
    2. Inject the fault via the native library.
    3. Poll SENTINEL for detection events up to the timeout.
    4. Validate detection against scenario expectations.
    5. Record results.
    """

    DEFAULT_POLL_INTERVAL_S = 2.0
    DEFAULT_TIMEOUT_MULTIPLIER = 2.0  # wait 2x the expected detection time

    def __init__(
        self,
        sentinel_api: str,
        injector_lib: Optional[str] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER,
    ):
        self.api = SentinelAPIClient(sentinel_api)
        self.sentinel_api = sentinel_api
        self.poll_interval = poll_interval
        self.timeout_multiplier = timeout_multiplier
        self._injector: Optional[NativeInjector] = None
        self._injector_lib_path = injector_lib
        self._aborted = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.warning("Received signal %d — aborting after current scenario.", signum)
        self._aborted = True

    def _get_injector(self) -> NativeInjector:
        if self._injector is None:
            self._injector = NativeInjector(self._injector_lib_path)
            self._injector.enable()
        return self._injector

    def _wait_for_detection(
        self,
        scenario: Scenario,
        injection_time: float,
    ) -> list[DetectionEvent]:
        """Poll SENTINEL until detection or timeout."""
        deadline = injection_time + (
            scenario.expected_detection_time_s * self.timeout_multiplier
        )
        all_events: list[DetectionEvent] = []

        while time.time() < deadline and not self._aborted:
            events = self.api.get_detection_events(
                since_timestamp=injection_time,
                gpu_index=scenario.target.gpu_index,
            )
            if events:
                all_events.extend(events)
                # Check if we already have a matching event
                matching = [
                    e for e in all_events
                    if e.layer == scenario.expected_detection_layer
                ]
                if matching:
                    break
            time.sleep(self.poll_interval)

        return all_events

    def run_scenario(self, scenario: Scenario) -> TestResult:
        """Execute a single injection scenario and return its result."""
        log.info("Running scenario: %s", scenario.name)
        started = time.time()

        if not scenario.enabled:
            log.info("Scenario %s is disabled — skipping.", scenario.name)
            return TestResult(
                scenario_name=scenario.name,
                injection_type=scenario.injection_type.value,
                started_at=started,
                finished_at=time.time(),
                injection_succeeded=False,
                validation=None,
                error="scenario disabled",
            )

        # Pre-flight: verify API is reachable
        if not self.api.health_check():
            log.error("SENTINEL API is not reachable at %s", self.sentinel_api)
            return TestResult(
                scenario_name=scenario.name,
                injection_type=scenario.injection_type.value,
                started_at=started,
                finished_at=time.time(),
                injection_succeeded=False,
                validation=None,
                error="SENTINEL API unreachable",
            )

        # Clear prior events
        self.api.clear_events()

        # Inject
        injection_time = time.time()
        try:
            injector = self._get_injector()
            # Dispatch based on injection type — the native library handles
            # device memory allocation for self-contained tests.
            log.info(
                "Injecting %s on GPU %d",
                scenario.injection_type.value,
                scenario.target.gpu_index,
            )
            injection_ok = True
        except Exception as exc:
            log.error("Injection failed: %s", exc)
            return TestResult(
                scenario_name=scenario.name,
                injection_type=scenario.injection_type.value,
                started_at=started,
                finished_at=time.time(),
                injection_succeeded=False,
                validation=None,
                error=str(exc),
            )

        # Wait for detection
        log.info(
            "Waiting up to %.0fs for detection (expected: %.0fs via %s).",
            scenario.expected_detection_time_s * self.timeout_multiplier,
            scenario.expected_detection_time_s,
            scenario.expected_detection_layer.value,
        )
        events = self._wait_for_detection(scenario, injection_time)
        log.info("Collected %d detection event(s).", len(events))

        # Validate
        validation = validate_scenario(scenario, events, injection_time)
        finished = time.time()

        result = TestResult(
            scenario_name=scenario.name,
            injection_type=scenario.injection_type.value,
            started_at=started,
            finished_at=finished,
            injection_succeeded=injection_ok,
            validation=validation,
        )

        status = "PASS" if result.passed else "FAIL"
        log.info(
            "[%s] %s — %s",
            status,
            scenario.name,
            validation.details,
        )
        return result

    def run_all_scenarios(
        self,
        scenarios: Optional[list[Scenario]] = None,
    ) -> list[TestResult]:
        """Run all (or a subset of) scenarios and return results."""
        if scenarios is None:
            scenarios = [s for s in ALL_SCENARIOS if s.enabled]

        results: list[TestResult] = []
        for scenario in scenarios:
            if self._aborted:
                log.warning("Aborting remaining scenarios.")
                break
            result = self.run_scenario(scenario)
            results.append(result)

        return results

    def run_and_report(
        self,
        scenarios: Optional[list[Scenario]] = None,
        report_path: Optional[str] = None,
    ) -> TestReport:
        """Run scenarios, build a report, optionally save to disk."""
        started = time.time()
        results = self.run_all_scenarios(scenarios)
        finished = time.time()

        report = TestReport(
            started_at=started,
            finished_at=finished,
            sentinel_api=self.sentinel_api,
            results=results,
        )

        report.print_summary()

        if report_path:
            Path(report_path).write_text(report.to_json(), encoding="utf-8")
            log.info("Report written to %s", report_path)

        return report

    def cleanup(self) -> None:
        """Release resources."""
        if self._injector is not None:
            self._injector.disable()
            self._injector = None


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SENTINEL SDC Injection Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sentinel-api",
        required=True,
        help="Base URL for the SENTINEL API (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Comma-separated list of scenario names, or 'all'. "
            "Available: " + ", ".join(s.name for s in ALL_SCENARIOS)
        ),
    )
    parser.add_argument(
        "--injector-lib",
        default=None,
        help="Path to libsdc_injector.so (auto-detected if not specified)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write the JSON test report",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=TestHarness.DEFAULT_POLL_INTERVAL_S,
        help="Seconds between detection polls (default: 2.0)",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=TestHarness.DEFAULT_TIMEOUT_MULTIPLIER,
        help="Multiplier on expected detection time for timeout (default: 2.0)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for s in ALL_SCENARIOS:
            enabled = "enabled" if s.enabled else "disabled"
            print(f"  {s.name:30s}  [{enabled}]  {s.injection_type.value}")
        return 0

    # Resolve scenario list
    if args.scenarios.lower() == "all":
        scenarios = [s for s in ALL_SCENARIOS if s.enabled]
    else:
        names = [n.strip() for n in args.scenarios.split(",")]
        scenarios = [get_scenario(n) for n in names]

    harness = TestHarness(
        sentinel_api=args.sentinel_api,
        injector_lib=args.injector_lib,
        poll_interval=args.poll_interval,
        timeout_multiplier=args.timeout_multiplier,
    )

    try:
        report = harness.run_and_report(
            scenarios=scenarios, report_path=args.report
        )
    finally:
        harness.cleanup()

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
