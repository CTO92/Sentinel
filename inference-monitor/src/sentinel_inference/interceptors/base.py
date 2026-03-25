"""Abstract base class for inference server output interceptors.

Every interceptor must implement ``capture_output`` which returns a
``TensorCapture`` containing the output tensor (as a NumPy array on host
memory) together with metadata describing the request.
"""

from __future__ import annotations

import abc
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TensorCapture:
    """Immutable snapshot of an intercepted inference output tensor.

    Attributes
    ----------
    tensor : np.ndarray
        Output logits / probabilities copied to host memory.  Shape is
        typically ``(vocab_size,)`` or ``(seq_len, vocab_size)``.
    timestamp : float
        Monotonic clock time (``time.monotonic()``) of capture.
    request_id : str
        Unique identifier for the inference request.
    input_hash : str
        SHA-256 hex digest of the serialised input tensor, used for
        cross-replica matching.
    model_name : str
        Name of the model that produced the output.
    gpu_id : int
        GPU ordinal that executed the inference.
    extra : dict[str, Any]
        Interceptor-specific metadata (e.g. Triton model version).
    """

    tensor: np.ndarray
    timestamp: float
    request_id: str
    input_hash: str
    model_name: str = ""
    gpu_id: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def compute_input_hash(data: bytes | np.ndarray) -> str:
    """Compute a SHA-256 hex digest used for cross-replica request matching."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha256(data).hexdigest()


class BaseInterceptor(abc.ABC):
    """Abstract interceptor for inference server output tensors.

    Subclasses hook into a specific inference runtime (Triton, vLLM, etc.)
    and implement ``capture_output`` which asynchronously yields captured
    tensor snapshots.
    """

    def __init__(self, *, gpu_id: int = 0, sample_rate: float = 0.01) -> None:
        self.gpu_id = gpu_id
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng(seed=None)

    def should_sample(self) -> bool:
        """Return True with probability ``sample_rate``."""
        return bool(self._rng.random() < self.sample_rate)

    @abc.abstractmethod
    async def capture_output(self) -> TensorCapture | None:
        """Capture one output tensor snapshot.

        Returns ``None`` if this inference was not sampled or if the
        capture failed gracefully.
        """
        ...

    @abc.abstractmethod
    async def start(self) -> None:
        """Perform any one-time setup (connect to shared memory, etc.)."""
        ...

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release resources."""
        ...

    def _make_capture(
        self,
        tensor: np.ndarray,
        request_id: str,
        input_hash: str,
        model_name: str = "",
        **extra: Any,
    ) -> TensorCapture:
        """Helper to build a ``TensorCapture`` with common fields filled."""
        return TensorCapture(
            tensor=np.ascontiguousarray(tensor, dtype=np.float32),
            timestamp=time.monotonic(),
            request_id=request_id,
            input_hash=input_hash,
            model_name=model_name,
            gpu_id=self.gpu_id,
            extra=extra,
        )
