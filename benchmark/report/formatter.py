"""Console + file output (SPEC §Headline Output, §Primary Result Table, §Machine-readable).

The console leads with the plain-language headline (determinism verdict + slowdown
factor + a "so what" wall-clock translation), then a compact breakdown table.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from datetime import datetime

from rich.console import Console
from rich.table import Table

_console = Console(width=140)


def _strict(mr):
    return next((r for r in mr.rows if r.preset == "strict"), None)


def _production(mr):
    return next((r for r in mr.rows if r.preset == "production"), None)


def _slowdown_str(row) -> str:
    if row is None or row.slowdown_factor is None:
        return "—"
    if abs(row.slowdown_factor - 1.0) < 0.05:
        return "1.0× (free)"
    return f"{row.slowdown_factor:.1f}× slower"


def _throughput_str(mr) -> str:
    prod, strict = _production(mr), _strict(mr)
    if prod is None or strict is None:
        return "—"
    if prod.model_class == "vision" and prod.throughput_img_s and strict.throughput_img_s:
        return f"{prod.throughput_img_s:,.0f} → {strict.throughput_img_s:,.0f} img/s"
    if prod.model_class == "llm":
        dp, ds = prod.extra.get("ttft_ms"), strict.extra.get("ttft_ms")
        if dp is not None and ds is not None:
            return f"first token +{ds - dp:.0f} ms"
    return "—"


def render_console(results, hardware: dict, cfg) -> None:
    _console.print()
    _console.rule("[bold]Can determinism be reached? And what does it cost? (production → strict)")

    table = Table(box=None, pad_edge=False)
    for col in ("Model", "Determinism", "Slowdown", "Throughput", "Memory"):
        table.add_column(col)

    for mr in results:
        label = f"{mr.spec.display_name}  ({mr.spec.model_class})"
        if mr.skipped:
            table.add_row(label, "[dim]skipped[/dim]", "—", "—", f"[dim]{mr.skip_reason}[/dim]")
            continue
        strict = _strict(mr)
        det = strict.determinism_achieved if strict else "n/a"
        det_color = {"YES": "green", "NO": "red", "UNSUPPORTED": "yellow"}.get(det, "white")
        achieved = det == "YES"
        slow = _slowdown_str(strict) if achieved else "—"
        thru = _throughput_str(mr) if achieved else "—"
        mem = (
            f"{strict.memory_overhead_mb:+.0f} MB"
            if achieved and strict and strict.memory_overhead_mb is not None
            else "—"
        )
        table.add_row(label, f"[{det_color}]{det}[/{det_color}]", slow, thru, mem)

    _console.print(table)
    _so_what(results)
    _console.rule()
    _primary_table(results)
    _notes(results)


def _so_what(results) -> None:
    """Translate the worst vision slowdown into wall-clock for a tangible workload."""
    candidates = []
    for mr in results:
        if mr.skipped:
            continue
        prod, strict = _production(mr), _strict(mr)
        if (
            prod and strict and prod.model_class == "vision"
            and prod.throughput_img_s and strict.throughput_img_s
            and strict.determinism_achieved == "YES"
        ):
            candidates.append((mr.spec.display_name, prod.throughput_img_s, strict.throughput_img_s))
    if not candidates:
        return
    name, p, s = max(candidates, key=lambda c: c[1] / c[2])
    n = 1_000_000
    prod_min, strict_min = (n / p) / 60.0, (n / s) / 60.0
    _console.print(
        f"\n  [bold]So what?[/bold] Serving {n:,} images on {name}:\n"
        f"    production ≈ {prod_min:.1f} min   strict ≈ {strict_min:.1f} min   "
        f"→  +{strict_min - prod_min:.1f} min for reproducibility"
    )


def _primary_table(results) -> None:
    _console.print("\n  [bold]Primary results[/bold]")
    t = Table(box=None, pad_edge=False)
    for col in ("Model", "Backend", "Preset", "Requested", "Effective",
                "Determinism", "Latency Cost", "Mem Cost", "Verdict"):
        t.add_column(col)
    for mr in results:
        if mr.skipped:
            continue
        for row in mr.rows:
            lat = "baseline" if row.preset == "production" else (
                f"{row.det_overhead_pct:+.1f}%" if row.det_overhead_pct is not None else "—"
            )
            mem = (
                f"{row.memory_overhead_mb:+.0f} MB"
                if row.memory_overhead_mb is not None else "—"
            )
            det = row.determinism_achieved if row.preset == "strict" else "—"
            t.add_row(
                row.model, row.backend, row.preset, row.requested_det_level,
                row.effective_det_level, det, lat, mem, row.verdict,
            )
    _console.print(t)


def _notes(results) -> None:
    for mr in results:
        for row in getattr(mr, "rows", []):
            if row.error:
                _console.print(f"  [yellow]Note:[/yellow] {row.model} {row.preset} — {row.error}")
    if any(getattr(mr, "rows", []) and mr.rows[0].backend == "mlx" for mr in results):
        _console.print(
            "  [dim]Note: MLX exposes no deterministic-algorithm controls "
            "(effective level = SEED_ONLY). Bit-exactness cannot be asserted on this backend.[/dim]"
        )
    _console.print(
        "  [dim]Determinism: YES = bit-identical every run on this hardware · "
        "NO = still varied · UNSUPPORTED = controls unavailable[/dim]"
    )


# --- file outputs -------------------------------------------------------

_CSV_FIELDS = [
    "model", "backend", "preset", "requested_det_level", "effective_det_level", "dtype",
    "determinism_achieved", "latency_p50_ms", "det_overhead_pct", "slowdown_factor",
    "memory_overhead_mb", "bit_exact_rate", "determinism_result",
]


def _row_dict(row) -> dict:
    d = dataclasses.asdict(row)
    d.update(d.pop("extra"))
    return d


def write_outputs(results, hardware: dict, cfg, output: str | None, formats: list[str],
                  label: str = "run") -> list[str]:
    want_json = "json" in formats
    want_csv = "csv" in formats
    if not (want_json or want_csv):
        return []

    if output:
        stem = os.path.splitext(output)[0]
    else:
        os.makedirs("results", exist_ok=True)
        stem = os.path.join("results", f"{label}_{datetime.now():%Y%m%d_%H%M%S}")

    written: list[str] = []
    all_rows = [_row_dict(r) for mr in results if not mr.skipped for r in mr.rows]

    if want_json:
        path = stem + ".json"
        payload = {
            "hardware": hardware,
            "config": {
                "tier": cfg.tier, "presets": cfg.presets, "dtype": cfg.dtype,
                "n_runs": cfg.n_runs, "n_warmup": cfg.n_warmup,
            },
            "skipped": [
                {"model": mr.spec.display_name, "reason": mr.skip_reason}
                for mr in results if mr.skipped
            ],
            "results": all_rows,
            "scope": "Empirical for this exact device/library stack only.",
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        written.append(path)

    if want_csv:
        path = stem + ".csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        written.append(path)

    if written:
        _console.print(f"\n  [dim]Results written: {' · '.join(written)}[/dim]")
    return written
