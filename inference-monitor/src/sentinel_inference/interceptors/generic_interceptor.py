"""Generic shared-memory tensor interceptor.

Captures tensors written to a named POSIX shared-memory region by any
CUDA process.  The producing process writes a simple header followed by
the raw float32 tensor data; this interceptor polls and copies sampled
snapshots for analysis.

Shared-memory layout (little-endian):
    uint32  magic       0x53_4E_54_4C ("SNTL")
    uint32  sequence    monotonically increasing write counter
    uint32  ndim        number of dimensions
    uint32[ndim] shape  tensor shape
    uint64  data_bytes  byte length of tensor payload
    float32[...]        tensor data
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

_MAGIC = 0x534E544C
# magic(4) + sequence(4) + ndim(4) = 12 bytes fixed prefix
_PREFIX_FMT = "<III"
_PREFIX_SIZE = struct.calcsize(_PREFIX_FMT)


class GenericShmInterceptor(BaseInterceptor):
    """Read output tensors from a named shared-memory region.

    Parameters
    ----------
    shm_name : str
        POSIX shared memory name.
    shm_size : int
        Expected size in bytes.
    poll_interval : float
        Seconds between polls.
    model_name : str
        Model name tag for captures.
    """

    def __init__(
        self,
        *,
        shm_name: str = "/sentinel_tensor_shm",
        shm_size: int = 64 * 1024 * 1024,
        model_name: str = "generic",
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
        try:
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
            logger.info("generic_interceptor_started", shm_name=self._shm_name)
        except FileNotFoundError:
            logger.warning(
                "generic_shm_not_found",
                shm_name=self._shm_name,
                msg="Will attempt lazy attach on first capture.",
            )

    async def stop(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        logger.info("generic_interceptor_stopped")

    async def capture_output(self) -> TensorCapture | None:
        if self._shm is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
            except FileNotFoundError:
                await asyncio.sleep(self._poll_interval)
                return None

        buf = self._shm.buf

        # Read prefix
        magic, seq, ndim = struct.unpack_from(_PREFIX_FMT, buf, 0)
        if magic != _MAGIC:
            await asyncio.sleep(self._poll_interval)
            return None

        # Skip if we already processed this sequence number
        if seq <= self._last_seq:
            await asyncio.sleep(self._poll_interval)
            return None

        self._last_seq = seq

        if not self.should_sample():
            return None

        offset = _PREFIX_SIZE
        if ndim == 0 or ndim > 8:
            logger.warning("generic_interceptor_bad_ndim", ndim=ndim)
            return None

        shape = struct.unpack_from(f"<{ndim}I", buf, offset)
        offset += 4 * ndim

        (data_bytes,) = struct.unpack_from("<Q", buf, offset)
        offset += 8

        num_elements = int(data_bytes // 4)
        expected_elements = 1
        for s in shape:
            expected_elements *= s
        if num_elements != expected_elements:
            logger.warning(
                "generic_interceptor_size_mismatch",
                data_bytes=data_bytes,
                expected=expected_elements * 4,
            )
            return None

        tensor = np.frombuffer(buf, dtype=np.float32, count=num_elements, offset=offset)
        tensor = tensor.reshape(shape).copy()

        input_hash = compute_input_hash(tensor)
        return self._make_capture(
            tensor=tensor,
            request_id=str(uuid.uuid4()),
            input_hash=input_hash,
            model_name=self._model_name,
            sequence=seq,
        )
