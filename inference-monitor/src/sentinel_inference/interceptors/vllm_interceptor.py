"""Interceptor for vLLM inference engine.

Hooks into vLLM's SamplerOutput via a plugin mechanism. The interceptor
registers a callback that receives output logits after each sampling step
and enqueues sampled copies for analysis.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import structlog

from sentinel_inference.interceptors.base import BaseInterceptor, TensorCapture, compute_input_hash

logger = structlog.get_logger(__name__)


class VLLMInterceptor(BaseInterceptor):
    """Captures output tensors from vLLM via SamplerOutput hook.

    This interceptor exposes a callback ``on_sampler_output`` that should be
    registered with vLLM's engine.  When vLLM calls the hook, the
    interceptor probabilistically samples the output, copies the tensor to
    host memory, and makes it available via ``capture_output``.

    Parameters
    ----------
    model_name : str
        Model name for metadata.
    queue_maxsize : int
        Internal buffer size for captured tensors.
    """

    def __init__(
        self,
        *,
        model_name: str = "vllm_model",
        queue_maxsize: int = 256,
        gpu_id: int = 0,
        sample_rate: float = 0.01,
    ) -> None:
        super().__init__(gpu_id=gpu_id, sample_rate=sample_rate)
        self._model_name = model_name
        self._queue: asyncio.Queue[TensorCapture] = asyncio.Queue(maxsize=queue_maxsize)
        self._hook_installed = False

    async def start(self) -> None:
        """Mark interceptor as ready.  Hook installation happens via
        ``get_hook`` which returns the callback for vLLM registration."""
        self._hook_installed = True
        logger.info("vllm_interceptor_started", model_name=self._model_name)

    async def stop(self) -> None:
        """Drain remaining items and shut down."""
        self._hook_installed = False
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("vllm_interceptor_stopped")

    def get_hook(self) -> Callable[..., None]:
        """Return a callback suitable for vLLM SamplerOutput registration.

        The returned callable has the signature::

            hook(request_id: str, input_ids: np.ndarray, logits: np.ndarray) -> None

        It is safe to call from any thread; it schedules work on the
        event loop.
        """

        def _hook(
            request_id: str,
            input_ids: np.ndarray,
            logits: np.ndarray,
            **kwargs: Any,
        ) -> None:
            if not self._hook_installed:
                return
            if not self.should_sample():
                return
            # Copy tensor to contiguous host memory immediately
            tensor_copy = np.array(logits, dtype=np.float32, copy=True)
            input_hash = compute_input_hash(input_ids)
            capture = self._make_capture(
                tensor=tensor_copy,
                request_id=request_id or str(uuid.uuid4()),
                input_hash=input_hash,
                model_name=self._model_name,
            )
            try:
                self._queue.put_nowait(capture)
            except asyncio.QueueFull:
                logger.warning("vllm_interceptor_queue_full", dropped_request=request_id)

        return _hook

    async def capture_output(self) -> TensorCapture | None:
        """Retrieve the next sampled tensor from the internal queue.

        Blocks until a capture is available or a short timeout expires.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except TimeoutError:
            return None
