from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np

from benchmark.backends.base import Backend


class CPUBackend(Backend):
    """PyTorch CPU backend. Reference / sanity-check path (SPEC §Hardware Backends).

    `torch.use_deterministic_algorithms(True)` is honored on CPU, so FULL_DET is
    enforceable; the cuDNN-specific knobs are simply no-ops here.
    """

    NAME = "cpu"

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def unavailable_reason() -> str:
        return "PyTorch not importable"

    @property
    def max_det_level(self) -> str:
        return "FULL_DET"

    def hardware_metadata(self) -> dict[str, Any]:
        import torch

        return {
            "device": "cpu",
            "torch_version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "mkl_available": torch.backends.mkl.is_available(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        }

    def time_call(self, fn: Callable[[], Any]) -> tuple[float, Any]:
        import torch

        with torch.no_grad():
            start = time.perf_counter()
            output = fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms, output

    def to_numpy(self, output: Any) -> np.ndarray:
        import torch

        if isinstance(output, torch.Tensor):
            return output.detach().to(torch.float32).cpu().numpy()
        return np.asarray(output, dtype=np.float32)

    # CPU peak memory is process-RSS based and coarse; reported as a reference only.
    def reset_peak_memory(self) -> None:
        self._rss0 = self._rss()

    def peak_memory_mb(self) -> float | None:
        base = getattr(self, "_rss0", None)
        cur = self._rss()
        if base is None or cur is None:
            return None
        return max(0.0, (cur - base) / (1024 * 1024))

    @staticmethod
    def _rss() -> float | None:
        try:
            import psutil
        except Exception:
            return None
        return float(psutil.Process().memory_info().rss)
