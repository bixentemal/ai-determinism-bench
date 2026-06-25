# AI Inference Determinism Benchmark

Measures **the cost of making neural-network inference bit-exact reproducible** on fixed
hardware. It answers two questions, in order:

1. **Can bit-exact determinism be reached?** — a plain `YES` / `NO` / `UNSUPPORTED` verdict, for
   anyone who must assert "this model produces identical output on every run on this hardware."
2. **What does it cost?** — the latency / throughput / memory penalty paid to get there, framed as a
   slowdown factor ("making this reproducible makes it 2× slower").

The headline contrast is two presets: **`production`** (the fast config real serving uses) vs
**`strict`** (bit-exact determinism, every control on). See [`SPEC.md`](SPEC.md) for the full design.

---

## Status

This is a **runnable vertical slice**, not the whole spec yet.

| Area | Implemented | Fast-follow |
|------|-------------|-------------|
| Backends | **CPU** (PyTorch), **MLX** (Apple Silicon) | **CUDA** (stubbed — needs a GPU host) |
| Vision | ResNet-50, ViT-B/16 | — |
| LLM | GPT-2 | Llama-3.2-1B |
| SSM | — | Mamba-370M |
| Determinism | `production` vs `strict` presets | `--expert` 4-level decomposition |

On Apple Silicon, ResNet-50 / ViT-B/16 / GPT-2 all run **MLX-native** (vision models are hand-ported
from torchvision weights and validated to match within fp tolerance). The **real determinism cost
tax requires the CUDA backend**, which is stubbed — on CPU and MLX, reproducibility is essentially
free or unassertable (see [Caveats](#caveats--scope)).

---

## Install

Uses [uv](https://docs.astral.sh/uv/) for reproducible, isolated installs (a `uv.lock` is committed):

```bash
uv sync --extra mlx          # isolated .venv with the Apple MLX backend
# or, CPU-only:
uv sync
```

`--extra mlx` adds `mlx` + `mlx-lm` for the Apple GPU backend; omit it on non-Apple machines.

---

## Usage

```bash
# Default: production vs strict on the core suite, headline first (auto-detects backend)
uv run python -m benchmark

# Fast smoke test (small shapes, few runs)
uv run python -m benchmark --quick

# Pick a backend explicitly
uv run python -m benchmark --backend cpu
uv run python -m benchmark --backend mlx

# Restrict to model classes
uv run python -m benchmark --models vision
uv run python -m benchmark --models llm

# Override precision (default: fp32 vision, bf16 LLM) and run counts
uv run python -m benchmark --dtype fp32 --n-runs 100 --n-warmup 20

# Write machine-readable output
uv run python -m benchmark --format table,json,csv --output results/run1
```

Results (JSON/CSV) land in `results/` (gitignored).

---

## What it measures

**Determinism presets** (the default axis):

| Preset | Meaning | Knobs |
|--------|---------|-------|
| `production` | fast, non-reproducible — what real serving runs | TF32 on, cuDNN autotune on, no deterministic algorithms |
| `strict` | bit-exact, every control on | TF32 off, deterministic algorithms, cuBLAS workspace pin |

**Per cell** (model × backend × preset): latency p50/p95/p99, peak memory, and *actual* numerical
reproducibility — `bit_exact_rate`, `output_max_abs_diff`, `output_std` (all vs the first run).
Vision adds `throughput_img_s`; LLM adds `ttft_ms`, `decode_tok_s`, `token_repro_rate`.

Each row records **`requested` vs `effective`** determinism level, so backend no-ops aren't misread
(e.g. MLX requests `strict` but can only enforce `SEED_ONLY`).

---

## Reading the output

The run leads with a plain-language headline:

```
  Model                 Determinism   Slowdown      Throughput            Memory
  ResNet-50  (vision)   YES           2.1× slower   4,200 → 1,950 img/s   +16 MB
  GPT-2      (LLM)      UNSUPPORTED   —             —                     —
```

- **`Determinism`** is the first answer — `YES` (bit-identical every run on this hardware) /
  `NO` (still varied) / `UNSUPPORTED` (controls unavailable, so it can't be asserted). It carries
  **no** cost information.
- **`Slowdown`** is the lead cost number (`strict_p50 / production_p50`), blank unless determinism
  was achieved.

A compact breakdown table and a derived per-row `Verdict` follow.

---

## Results

**MLX backend, Apple Silicon** (`mlx 0.31.2`, Metal GPU; core tier, N=50 runs, 10 warmup;
fp32 vision / bf16 LLM). Generated with `uv run python -m benchmark --backend mlx`:

```
Can determinism be reached? And what does it cost? (production → strict)
Model                Determinism  Slowdown  Throughput  Memory
ResNet-50  (vision)  UNSUPPORTED  —         —           —
ViT-B/16  (vision)   UNSUPPORTED  —         —           —
GPT-2  (llm)         UNSUPPORTED  —         —           —

Primary results
Model      Backend  Preset      Requested  Effective  Determinism  Latency Cost  Mem Cost  Verdict
ResNet-50  mlx      production  NONE       NONE       —            baseline      +0 MB     Deterministic, low cost
ResNet-50  mlx      strict      FULL_DET   SEED_ONLY  UNSUPPORTED  +0.2%         +0 MB     Full controls unavailable
ViT-B/16   mlx      production  NONE       NONE       —            baseline      +0 MB     Deterministic, low cost
ViT-B/16   mlx      strict      FULL_DET   SEED_ONLY  UNSUPPORTED  +1.5%         +0 MB     Full controls unavailable
GPT-2      mlx      production  NONE       NONE       —            baseline      +0 MB     Deterministic, low cost
GPT-2      mlx      strict      FULL_DET   SEED_ONLY  UNSUPPORTED  +0.7%         -0 MB     Full controls unavailable
```

All three models are **empirically bit-exact** across runs in both presets, but MLX exposes no
deterministic-algorithm controls (`strict` requests `FULL_DET`, only `SEED_ONLY` is enforceable), so
determinism reports **`UNSUPPORTED`** — it's observed, not assertable. The latency delta between
presets is within noise (≤1.5%): on this backend reproducibility is effectively free, and the real
determinism cost story awaits the CUDA backend.

## Caveats & scope

`determinism_achieved = YES` is **empirical and scoped to the exact recorded stack** (GPU/CPU,
driver, library versions, dtype). It means "bit-identical every run on *this* hardware" — **not** a
guarantee across different machines, microarchitectures (SIMD width), library versions, or thread
counts. Every result row records the stack, and output is labelled "empirical for this exact stack
only."

Specifically:

- **CPU** reproducibility is robust here but is *not* a universal property of "CPU." It held across
  thread counts and processes for these models on this machine, but can differ across x86 vs ARM,
  BLAS/oneDNN versions, or for ops with no fixed reduction order.
- **MLX** exposes no deterministic-algorithm controls, so `strict` reports `UNSUPPORTED` even when
  output is empirically stable — bit-exactness cannot be *asserted*.
- The largest determinism cost (the "2× slower" story) lives on **CUDA**, which is stubbed in this
  slice.

---

## Project layout

```
benchmark/
├── cli.py / __main__.py        # python -m benchmark
├── config.py                   # run configuration
├── backends/   base · cpu · mlx · cuda(stub)
├── determinism/ modes · verify # presets + requested/effective level + verdicts
├── models/     vision · llm · _mlx_resnet · _mlx_vit
├── metrics/    timing · numerical
├── runners/    base · vision · llm
└── report/     aggregator · formatter   # headline + table + json/csv
```

Full design, metric definitions, and rationale: [`SPEC.md`](SPEC.md).
