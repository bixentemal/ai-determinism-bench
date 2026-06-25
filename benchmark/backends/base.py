from __future__ import annotations

import abc
from typing import Any, Callable

import numpy as np

# Determinism levels, ordered weakest -> strongest (SPEC §Determinism Levels).
DET_LEVELS = ["NONE", "SEED_ONLY", "CUDNN_DET", "FULL_DET"]


def clamp_level(requested: str, ceiling: str) -> str:
    """Downgrade a requested determinism level to what a backend can enforce."""
    return requested if DET_LEVELS.index(requested) <= DET_LEVELS.index(ceiling) else ceiling


class Backend(abc.ABC):
    """Abstract hardware backend.

    A backend owns three concerns the rest of the pipeline must not special-case:
    device-correct timing, peak-memory accounting, and how strong a determinism
    level it can actually enforce (`max_det_level`).
    """

    NAME: str = "base"

    # --- availability ---------------------------------------------------
    @staticmethod
    def available() -> bool:  # pragma: no cover - overridden
        return False

    @staticmethod
    def unavailable_reason() -> str:
        return "not implemented"

    @property
    @abc.abstractmethod
    def max_det_level(self) -> str:
        """Strongest determinism level this backend can enforce (SPEC: effective level)."""

    @abc.abstractmethod
    def hardware_metadata(self) -> dict[str, Any]:
        """Stack identification embedded in every result row (SPEC §Hardware Detection)."""

    # --- timing / materialization --------------------------------------
    @abc.abstractmethod
    def time_call(self, fn: Callable[[], Any]) -> tuple[float, Any]:
        """Run `fn`, force the result to be materialized, return (elapsed_ms, output)."""

    @abc.abstractmethod
    def to_numpy(self, output: Any) -> np.ndarray:
        """Convert a backend tensor to a host numpy array for numerical comparison."""

    @property
    def device(self) -> str:
        """PyTorch device string used for model and input placement."""
        return "cpu"

    # --- memory ---------------------------------------------------------
    def reset_peak_memory(self) -> None:
        return None

    def peak_memory_mb(self) -> float | None:
        return None
