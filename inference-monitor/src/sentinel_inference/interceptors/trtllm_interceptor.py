"""Interceptor for NVIDIA TensorRT-LLM inference engine.

Registers an output callback with TensorRT-LLM's executor API to capture
output logit tensors after each generation step.  The callback copies
sampled tensors to host memory for offline analysis.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import numpy as np
import structlog

from sentinel_inference.interceptors.base import BaseInterceptor, TensorCapture, compute_input_hash

logger = structlog.get_logger(__name__)


class TrtLLMInterceptor(BaseInterceptor):
    """Captures output tensors from TensorRT-LLM via output callback.

    TensorRT-LLM's executor supports registering callbacks that fire after
    each generation step.  This interceptor provides ``register_callback``
    to install its hook, then yields captures through ``capture_output``.

    Parameters
    ----------
    model_name : str
        Model name for metadata tags.
    queue_maxsize : int
        Internal buffer for captured tensors awaiting analysis.
    """

    def __init__(
        self,
        *,
        model_name: str = "trtllm_model",
        queue_maxsize: int = 256,
        gpu_id: int = 0,
        sample_rate: float = 0.01,
    ) -> None:
        super().__init__(gpu_id=gpu_id, sample_rate=sample_rate)
        self._model_name = model_name
        self._queue: asyncio.Queue[TensorCapture] = asyncio.Queue(maxsize=queue_maxsize)
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("trtllm_interceptor_started", model_name=self._model_name)

    async def stop(self) -> None:
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("trtllm_interceptor_stopped")

    def output_callback(
        self,
        request_id: str | None,
        input_token_ids: np.ndarray | None,
        output_logits: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Callback to register with TensorRT-LLM executor.

        This is called synchronously from the TRT-LLM runtime whenever a
        generation step completes.  It must return quickly.

        Parameters
        ----------
        request_id : str or None
            TRT-LLM request identifier.
        input_token_ids : np.ndarray or None
            Input token IDs for cross-replica hashing.
        output_logits : np.ndarray
            Raw output logits from the current step.
        """
        if not self._running:
            return
        if not self.should_sample():
            return

        tensor_copy = np.array(output_logits, dtype=np.float32, copy=True)
        if input_token_ids is not None:
            input_hash = compute_input_hash(input_token_ids)
        else:
            input_hash = compute_input_hash(tensor_copy)

        rid = request_id or str(uuid.uuid4())
        capture = self._make_capture(
            tensor=tensor_copy,
            request_id=rid,
            input_hash=input_hash,
            model_name=self._model_name,
        )
        try:
            self._queue.put_nowait(capture)
        except asyncio.QueueFull:
            logger.warning("trtllm_interceptor_queue_full", dropped_request=rid)

    async def capture_output(self) -> TensorCapture | None:
        """Retrieve the next captured tensor from the internal queue."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except TimeoutError:
            return None
