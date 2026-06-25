"""Actual numerical non-determinism across runs (SPEC §Universal metrics).

All comparisons are against `run[0]` (the SPEC reference-run rule), so this is
O(N) in runs and O(1) in stored tensors: we keep run[0] plus running Welford
mean/M2 accumulators rather than all N outputs (50 GPT-2 logit tensors would be
several GB).
"""

from __future__ import annotations

import numpy as np


class RunAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self._ref: np.ndarray | None = None  # run[0], raw dtype, for bit-exact + diff
        self._mean: np.ndarray | None = None  # float64, Welford
        self._m2: np.ndarray | None = None
        self._bit_exact = 0
        self._max_abs = 0.0

    def update(self, output: np.ndarray) -> None:
        raw = np.asarray(output)
        x = raw.astype(np.float64)
        self.n += 1
        if self._ref is None:
            self._ref = raw.copy()
            self._mean = x.copy()
            self._m2 = np.zeros_like(x)
            self._bit_exact = 1
            return

        if np.array_equal(raw, self._ref):
            self._bit_exact += 1
        diff = np.abs(x - self._ref.astype(np.float64))
        self._max_abs = max(self._max_abs, float(diff.max()))

        # Welford online update (per element).
        delta = x - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (x - self._mean)

    def finalize(self) -> dict[str, float]:
        if self.n == 0:
            return {"output_std": 0.0, "output_max_abs_diff": 0.0, "bit_exact_rate": 0.0}
        if self.n > 1:
            std = np.sqrt(self._m2 / (self.n - 1))
            output_std = float(std.mean())
        else:
            output_std = 0.0
        return {
            "output_std": output_std,
            "output_max_abs_diff": self._max_abs,
            "bit_exact_rate": 100.0 * self._bit_exact / self.n,
        }
