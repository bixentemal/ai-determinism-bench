"""Command-line entry point (SPEC §CLI Design).

Default run: production vs strict on the core tier, headline first. `--quick` is a
fast smoke profile. The granular four-level `--expert` sweep is fast-follow; this
slice runs the two-point preset contrast.
"""

from __future__ import annotations

import argparse

from benchmark.backends import get_backend, available_backends
from benchmark.config import RunConfig
from benchmark.models import select_models
from benchmark.report import finalize, render_console, write_outputs
from benchmark.runners import run_model


def _parse(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__)
    p.add_argument("--quick", action="store_true", help="fast smoke profile (small shapes, few runs)")
    p.add_argument("--expert", action="store_true", help="(fast-follow) granular 4-level decomposition")
    p.add_argument("--tier", choices=["quick", "core"], help="model tier (default: core)")
    p.add_argument("--backend", choices=["cpu", "mlx", "cuda"], help="backend (default: auto-detect)")
    p.add_argument("--models", help="comma list of classes: vision,llm")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], help="override dtype uniformly")
    p.add_argument("--n-runs", type=int, help="measurement runs")
    p.add_argument("--n-warmup", type=int, help="warmup runs")
    p.add_argument("--output", help="output path stem")
    p.add_argument("--format", default="table", help="comma list: table,json,csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    cfg = RunConfig.quick() if args.quick else RunConfig()

    if args.tier:
        cfg.tier = args.tier
    if args.models:
        cfg.classes = [c.strip() for c in args.models.split(",") if c.strip()]
    if args.dtype:
        cfg.dtype = {k: args.dtype for k in cfg.dtype}
    if args.n_runs is not None:
        cfg.n_runs = args.n_runs
    if args.n_warmup is not None:
        cfg.n_warmup = args.n_warmup
    cfg.output = args.output
    cfg.formats = [f.strip() for f in args.format.split(",") if f.strip()]

    if args.expert:
        print("Note: --expert (4-level decomposition) is fast-follow; running production vs strict.")

    backend = get_backend(args.backend)
    hardware = backend.hardware_metadata()
    specs = select_models(cfg.tier, cfg.classes)

    print("\nAI Determinism Benchmark — measuring the cost of reproducibility")
    print(f"Backend: {backend.NAME} · {hardware}")
    print(f"Available backends: {', '.join(available_backends())}")
    print(f"Tier: {cfg.tier} · dtype: {cfg.dtype} · N={cfg.n_runs} runs, {cfg.n_warmup} warmup")
    print("Comparing: production (fast)  vs  strict (bit-exact)\n")

    results = []
    n = len(specs)
    for i, spec in enumerate(specs, 1):
        print(f"  [{i}/{n}] {spec.display_name:<12} {spec.model_class}", flush=True)
        mr = run_model(spec, backend, cfg)
        if mr.skipped:
            print(f"        skipped: {mr.skip_reason}")
        results.append(mr)

    finalize(results, cfg)
    render_console(results, hardware, cfg)
    write_outputs(results, hardware, cfg, cfg.output, cfg.formats, label=backend.NAME)
    return 0
