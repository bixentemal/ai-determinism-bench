from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from benchmark.backends.base import Backend


class MLXBackend(Backend):
    """Apple MLX backend. Metal exposes no deterministic-algorithm controls, so the
    strongest enforceable level is SEED_ONLY — bit-exactness cannot be *asserted*
    here even when output happens to be stable (SPEC §Determinism Levels, MLX notes).

    MLX uses lazy evaluation, so timing must wrap an explicit `mx.eval` or it would
    measure graph construction, not execution (SPEC §Statistical Protocol).
    """

    NAME = "mlx"

    @staticmethod
    def available() -> bool:
        try:
            import mlx.core  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def unavailable_reason() -> str:
        return "mlx not importable (Apple Silicon only)"

    @property
    def max_det_level(self) -> str:
        return "SEED_ONLY"

    def hardware_metadata(self) -> dict[str, Any]:
        import mlx.core as mx

        return {
            "device": str(mx.default_device()),
            "mlx_version": mx.__version__,
            "metal": True,
        }

    def time_call(self, fn: Callable[[], Any]) -> tuple[float, Any]:
        import mlx.core as mx

        start = time.perf_counter()
        output = fn()
        mx.eval(output)  # force materialization before stopping the clock
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms, output

    def to_numpy(self, output: Any) -> np.ndarray:
        import mlx.core as mx

        if isinstance(output, mx.array):
            return np.asarray(output.astype(mx.float32))
        return np.asarray(output, dtype=np.float32)

    def reset_peak_memory(self) -> None:
        import mlx.core as mx

        reset = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
        if reset is not None:
            try:
                reset()
            except Exception:
                pass

    def peak_memory_mb(self) -> float | None:
        import mlx.core as mx

        get = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
        try:
            return get() / (1024 * 1024) if get is not None else None
        except Exception:
            return None
