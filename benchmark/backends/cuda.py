from __future__ import annotations

from typing import Any, Callable

import numpy as np

from benchmark.backends.base import Backend


class CUDABackend(Backend):
    """CUDA backend — STUB.

    The full implementation (cuDNN/cuBLAS determinism flags, `torch.cuda.Event`
    timing, `torch.cuda.max_memory_allocated`) is deferred until it can be run and
    validated on an actual NVIDIA host. On non-CUDA machines this backend simply
    reports itself unavailable so auto-detection skips it cleanly. See SPEC
    §Hardware Backends and §Statistical Protocol for the target behavior.

    To implement: time_call() uses cuda.Event start/end + synchronize; metadata
    reads get_device_name/properties + torch.version.cuda + cudnn.version();
    max_det_level is FULL_DET.
    """

    NAME = "cuda"

    @staticmethod
    def available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def unavailable_reason() -> str:
        return "torch.cuda.is_available() is False on this host"

    @property
    def max_det_level(self) -> str:
        return "FULL_DET"

    def hardware_metadata(self) -> dict[str, Any]:  # pragma: no cover - needs GPU
        raise NotImplementedError("CUDA backend is a stub; run on a CUDA host to implement.")

    def time_call(self, fn: Callable[[], Any]) -> tuple[float, Any]:  # pragma: no cover
        raise NotImplementedError("CUDA backend is a stub; run on a CUDA host to implement.")

    def to_numpy(self, output: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("CUDA backend is a stub; run on a CUDA host to implement.")
