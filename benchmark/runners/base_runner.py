"""Shared run loop: warmup, measure, accumulate (SPEC §Runners / §Statistical Protocol).

For each (model, backend, preset) cell: enter the determinism context, warm up,
run N timed measurement runs against one fixed input, accumulate numerical
reproducibility online, then add class-specific metrics. Cross-preset deltas
(overhead, slowdown) are computed later in the aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmark.determinism import PRESET_TO_LEVEL, determinism
from benchmark.metrics import RunAccumulator, summarize_latencies
from benchmark.models import ModelUnavailable, ModelSpec, load_cell
from benchmark.determinism.verify import classify_determinism_result, determinism_achieved
from benchmark.runners import llm_runner, vision_runner

_EXTRA = {"vision": vision_runner.measure, "llm": llm_runner.measure}


@dataclass
class CellRow:
    model: str
    model_class: str
    backend: str
    preset: str
    dtype: str
    requested_det_level: str
    effective_det_level: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    warmup_latency_ms: float
    memory_peak_mb: float | None
    output_std: float
    output_max_abs_diff: float
    bit_exact_rate: float
    determinism_result: str
    determinism_achieved: str
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Filled by the aggregator (cross-preset deltas):
    det_overhead_pct: float | None = None
    memory_overhead_mb: float | None = None
    slowdown_factor: float | None = None
    throughput_img_s: float | None = None
    verdict: str = ""


@dataclass
class ModelResult:
    spec: ModelSpec
    backend: str
    rows: list[CellRow] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def _run_one(cell, backend, preset: str, inp, cfg) -> CellRow:
    requested_level = PRESET_TO_LEVEL[preset]
    error: str | None = None
    warmup_latency_ms = float("nan")
    acc = RunAccumulator()
    task_ref: np.ndarray | None = None
    task_stable_sig = True
    has_task_sig = hasattr(cell, "task_signature")
    latencies: list[float] = []
    extra: dict[str, Any] = {}
    task_stable_override: bool | None = None

    with determinism(preset, backend, cfg.seed) as effective_level:
        try:
            for i in range(cfg.n_warmup):
                ms, _ = backend.time_call(lambda: cell.infer(inp))
                if i == 0:
                    warmup_latency_ms = ms

            backend.reset_peak_memory()
            for _ in range(cfg.n_runs):
                ms, out = backend.time_call(lambda: cell.infer(inp))
                latencies.append(ms)
                np_out = backend.to_numpy(out)
                acc.update(np_out)
                if has_task_sig:
                    sig = cell.task_signature(np_out)
                    if task_ref is None:
                        task_ref = sig
                    elif not np.array_equal(sig, task_ref):
                        task_stable_sig = False

            extra, task_stable_override = _EXTRA[cell.model_class](cell, backend, inp, cfg)
        except RuntimeError as e:
            # e.g. FULL_DET rejected a nondeterministic op (SPEC: determinism_result=ERROR)
            error = str(e)

    lat_stats = summarize_latencies(latencies) if latencies else {
        "latency_p50_ms": float("nan"),
        "latency_p95_ms": float("nan"),
        "latency_p99_ms": float("nan"),
    }
    num = acc.finalize()
    task_stable = task_stable_override if task_stable_override is not None else (
        task_stable_sig if has_task_sig else None
    )

    det_result = classify_determinism_result(
        bit_exact_rate=num["bit_exact_rate"],
        task_stable=task_stable,
        effective_level=effective_level,
        requested_level=requested_level,
        error=error,
    )
    achieved = (
        determinism_achieved(
            effective_level=effective_level,
            bit_exact_rate=num["bit_exact_rate"],
            error=error,
        )
        if preset == "strict"
        else "n/a"
    )

    return CellRow(
        model=cell.display_name,
        model_class=cell.model_class,
        backend=backend.NAME,
        preset=preset,
        dtype=cell.dtype_str,
        requested_det_level=requested_level,
        effective_det_level=effective_level,
        warmup_latency_ms=warmup_latency_ms,
        memory_peak_mb=backend.peak_memory_mb(),
        output_std=num["output_std"],
        output_max_abs_diff=num["output_max_abs_diff"],
        bit_exact_rate=num["bit_exact_rate"],
        determinism_result=det_result,
        determinism_achieved=achieved,
        extra=extra,
        error=error,
        **lat_stats,
    )


def run_model(spec: ModelSpec, backend, cfg, progress=None) -> ModelResult:
    dtype = cfg.dtype_for(spec.model_class)
    try:
        cell = load_cell(spec, backend, dtype, cfg)
    except ModelUnavailable as e:
        return ModelResult(spec, backend.NAME, skipped=True, skip_reason=str(e))
    except Exception as e:  # model load failure shouldn't kill the whole run
        return ModelResult(spec, backend.NAME, skipped=True, skip_reason=f"load failed: {e}")

    inp = cell.make_input()
    rows: list[CellRow] = []
    for preset in cfg.presets:
        if progress:
            progress(spec, preset)
        rows.append(_run_one(cell, backend, preset, inp, cfg))
    return ModelResult(spec, backend.NAME, rows=rows, extra_metadata=cell.extra_metadata())
