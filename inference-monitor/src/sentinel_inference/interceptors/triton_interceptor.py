"""Interceptor for NVIDIA Triton Inference Server.

Hooks into Triton's shared-memory output cache to capture output tensors
without modifying the inference pipeline. Uses the Triton client library
to register a shared memory region and poll for completed inferences.
"""

from __future__ import annotations

import asyncio
import struct
import uuid
from multiprocessing import shared_memory

import numpy as np
import structlog

from sentinel_inference.interceptors.base import BaseInterceptor, TensorCapture, compute_input_hash

logger = structlog.get_logger(__name__)

# Triton shared-memory header layout (little-endian):
#   uint32 magic (0x5452_4954 = "TRIT")
#   uint32 flags  (bit 0 = ready)
#   uint64 tensor_byte_size
#   uint32 ndim
#   uint32[ndim] shape
#   float32[...] data
_HEADER_MAGIC = 0x54524954
_HEADER_FMT = "<IIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class TritonInterceptor(BaseInterceptor):
    """Captures output tensors from Triton Inference Server shared memory.

    Parameters
    ----------
    shm_name : str
        Name of the POSIX shared-memory region registered with Triton for
        output tensor caching.
    shm_size : int
        Size of the shared-memory region in bytes.
    model_name : str
        Model name to tag captures with.
    poll_interval : float
        Seconds between polling attempts.
    """

    def __init__(
        self,
        *,
        shm_name: str = "/sentinel_triton_output",
        shm_size: int = 64 * 1024 * 1024,
        model_name: str = "triton_model",
        poll_interval: float = 0.001,
        gpu_id: int = 0,
        sample_rate: float = 0.01,
    ) -> None:
        super().__init__(gpu_id=gpu_id, sample_rate=sample_rate)
        self._shm_name = shm_name
        self._shm_size = shm_size
        self._model_name = model_name
        self._poll_interval = poll_interval
        self._shm: shared_memory.SharedMemory | None = None
        self._last_seq: int = 0

    async def start(self) -> None:
        """Attach to the Triton output shared-memory region."""
        try:
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
            logger.info(
                "triton_interceptor_started",
                shm_name=self._shm_name,
                shm_size=self._shm_size,
            )
        except FileNotFoundError:
            logger.warning(
                "triton_shm_not_found",
                shm_name=self._shm_name,
                msg="Will retry on first capture attempt.",
            )

    async def stop(self) -> None:
        """Detach from shared memory."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        logger.info("triton_interceptor_stopped")

    async def capture_output(self) -> TensorCapture | None:
        """Poll shared memory for a new completed output tensor."""
        if self._shm is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
            except FileNotFoundError:
                await asyncio.sleep(self._poll_interval)
                return None

        buf = self._shm.buf

        # Read header
        magic, flags, tensor_bytes = struct.unpack_from(_HEADER_FMT, buf, 0)
        if magic != _HEADER_MAGIC:
            await asyncio.sleep(self._poll_interval)
            return None

        ready = bool(flags & 1)
        if not ready:
            await asyncio.sleep(self._poll_interval)
            return None

        if not self.should_sample():
            # Clear ready flag so the next inference can write
            struct.pack_into("<I", buf, 4, flags & ~1)
            return None

        offset = _HEADER_SIZE
        (ndim,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        shape = struct.unpack_from(f"<{ndim}I", buf, offset)
        offset += 4 * ndim

        num_elements = 1
        for s in shape:
            num_elements *= s

        tensor = np.frombuffer(
            buf,
            dtype=np.float32,
            count=num_elements,
            offset=offset,
        ).reshape(shape).copy()

        # Clear ready flag
        struct.pack_into("<I", buf, 4, flags & ~1)

        input_hash = compute_input_hash(tensor)
        request_id = str(uuid.uuid4())

        return self._make_capture(
            tensor=tensor,
            request_id=request_id,
            input_hash=input_hash,
            model_name=self._model_name,
        )
