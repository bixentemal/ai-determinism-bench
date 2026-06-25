"""Determinism modes as context managers (SPEC §Determinism Levels / §Presets).

Granular levels, weakest -> strongest, each a superset of the previous:

    NONE < SEED_ONLY < CUDNN_DET < FULL_DET

The two headline presets bundle these for the default two-point cost story:

    production -> NONE  + fast production knobs (TF32 ON, cuDNN autotune ON)
    strict     -> FULL_DET (TF32 OFF, deterministic algorithms, cuBLAS workspace pin)

Knobs that don't apply to the active backend are no-ops (e.g. cuDNN flags on CPU/MLX).
The manager yields the *effective* level it could actually enforce, which the runner
records alongside the requested one so backend no-ops aren't misread as bugs.
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np

from benchmark.backends.base import clamp_level

PRESET_TO_LEVEL = {"production": "NONE", "strict": "FULL_DET"}


def preset_label(spec: str) -> str:
    return {"production": "production", "strict": "strict"}.get(spec, spec)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except Exception:
        pass


def _set_torch_flags(level: str, fast: bool) -> None:
    """Apply backend determinism knobs. cuDNN/cuBLAS knobs are inert off CUDA but
    are set uniformly so the same code path runs everywhere (SPEC §Presets)."""
    try:
        import torch
    except Exception:
        return

    tf32 = fast  # TF32 is folded into the headline cost: ON in production, OFF in strict.
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    except Exception:
        pass

    if level in ("CUDNN_DET", "FULL_DET"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = fast  # autotune ON in production

    if level == "FULL_DET":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


@contextlib.contextmanager
def determinism(spec: str, backend, seed: int = 1234):
    """Enter the determinism mode named by `spec` (a preset or a granular level).

    Yields the effective level (clamped to what `backend` can enforce). The
    requested level is the caller's `spec`; the yielded value is the effective one.
    """
    is_preset = spec in PRESET_TO_LEVEL
    requested_level = PRESET_TO_LEVEL.get(spec, spec)
    effective = clamp_level(requested_level, backend.max_det_level)
    fast = (spec == "production")

    if effective != "NONE":
        _seed_all(seed)
    _set_torch_flags(effective, fast=fast)
    try:
        yield effective
    finally:
        # Leave deterministic-algorithm enforcement off between cells so a strict
        # cell can't make a later production cell raise on a nondeterministic op.
        try:
            import torch

            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
