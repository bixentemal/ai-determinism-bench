# AI Inference Determinism Benchmark — SPEC

## Goal

This benchmark answers **two questions, in order**, for a given (model, batch) on **fixed
hardware**:

1. **Can bit-exact determinism actually be reached?** A plain `YES` / `NO` / `UNSUPPORTED` verdict —
   for safety-constrained users who must be able to assert "this model produces identical output on
   every run on this hardware." This is the first column of every output and carries **no** cost
   information.
2. **What does it cost?** The latency/throughput/memory delta paid to get there — framed so a
   non-technical reader can *feel* it ("making this reproducible makes it 2× slower").

We do not care about raw absolute performance. We care about ratios — so HuggingFace model
wrappers are acceptable overhead since they cancel out in the delta.

**Audience and intent.** Two audiences, one tool. (a) *Safety / reliability* users who need a
hardware-pinned yes/no on reproducibility before they can ship a model into a constrained
environment. (b) A largely *non-technical* audience who complain that neural networks are
non-deterministic and assume reproducibility is free — for them the headline contrast is
deliberately stark: **the fast config real production uses** (`production`) versus **ultra-strict
bit-exact determinism** (`strict`), with the cost shown as a slowdown factor, especially for vision
models and LLMs where the tax is largest. Technical depth (four-level decomposition, per-knob
isolation) is preserved behind `--expert`, but the default experience leads with the reachability
verdict, then the two-point cost story, in plain language.

Design drivers:
- Representativity across neural network types and architectures
- Simple one-command usage on common hardware
- Simple result interpretation: a non-technical reader should grasp the cost in one sentence
  (slowdown factor + a tangible "so what"), with full detail available on demand

---

## Scope

### Model Classes

| Class   | Representative models         | Why these |
|---------|-------------------------------|-----------|
| Vision  | ResNet-50, ViT-B/16           | CNNs and attention, different op mix |
| LLM     | GPT-2 (always), Llama-3.2-1B  | GPT-2 fits anywhere; Llama exercises real decode at scale |
| SSM     | Mamba-370M                    | Exposes recurrent vs parallel mode split |

All loaded via HuggingFace `transformers` / `mamba-ssm`. Rationale: HF adds a fixed overhead
that is present in all determinism levels equally — it does not distort the delta we measure.

#### Why each model — the non-determinism mechanism it isolates

The models are not chosen for popularity; each one stresses a *different dominant source* of
floating-point non-determinism, so together they cover the space of reasons GPU inference is
irreproducible. The determinism cost is paid wherever a parallel reduction must be made
order-stable, and each architecture concentrates that reduction in a different place.

| Model | Dominant non-determinism source it isolates |
|-------|---------------------------------------------|
| **ResNet-50** (CNN) | **Convolution.** cuDNN picks among many conv algorithms (Winograd, FFT, implicit GEMM) by autotuning; each has a different reduction tree, so the *autotuner itself* is a source of run-to-run variance. This is the model that exposes the cost of disabling autotuning + forcing deterministic conv kernels — expected to be the single largest overhead. |
| **ViT-B/16** (vision transformer) | **Massive softmax / attention reduction.** Same image task as ResNet but no convolutions — the work is large matmuls and softmax over long attention rows. It isolates the cost of order-stable reductions in attention without conv autotuning, and contrasts directly with ResNet on the *same* input to show CNN-cost vs attention-cost. |
| **GPT-2** (decoder LLM) | **Attention kernel selection (SDPA/FlashAttention) + GEMM, split prefill vs decode.** Prefill is a big parallel attention reduction (FlashAttention's reduction order is the classic non-determinism culprit); decode is memory-bandwidth bound over the KV cache. Separating the two shows determinism is expensive in prefill but nearly free in decode. |
| **Llama-3.2-1B** (decoder LLM, extended) | Same mechanisms as GPT-2 but at **realistic scale and BF16** — confirms the prefill/decode finding holds for a model people actually serve, where larger GEMMs and longer contexts amplify the reduction cost. |
| **Mamba-370M** (SSM) | **Parallel scan vs sequential recurrence.** Prefill uses a parallel associative scan (order-sensitive, like attention); decode is a sequential recurrence that is deterministic *by construction*. It is the control case showing an architecture where determinism is free in one mode and costly in the other. |

Why two vision models specifically: ResNet and ViT run the **identical input and task** but route
the compute through convolution vs attention. Holding everything else constant isolates "cost of
making convolution deterministic" from "cost of making attention deterministic" — the single
clearest comparison in the suite.

### Precision (dtype)

Precision is fixed per model class and recorded on every result row. This is not optional: dtype
sets the *magnitude* of FP non-associativity — the exact effect this benchmark measures — so a
single dtype choice is the only way the numerical metrics are comparable within a model class.

| Model class | Default dtype | Why |
|-------------|---------------|-----|
| Vision      | FP32          | Reference precision; isolates the determinism effect from precision loss |
| LLM         | BF16          | Real-world serving precision for decoder LLMs |
| SSM         | BF16          | Matches LLM serving conditions |

Overridable with `--dtype {fp32,bf16,fp16}` (applied uniformly). Mixing dtypes across model classes
in a single comparison is intentional; the delta is always measured *within* a (model, backend,
dtype) cell, never across dtypes.

**TF32** is bundled into the determinism presets (see §Determinism Presets), not held constant.
It is **ON** in `production` (the fast path real serving uses) and **OFF** in `strict`:
```python
# strict only:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```
This intentionally folds TF32 (a precision lever) into the headline cost. The goal of this
benchmark is to show the *real* cost of going from a normal production config to bit-exact
determinism, and TF32 is part of what real production uses — neutralizing it everywhere would
understate the tax. `tf32_enabled` is recorded per row so the choice is auditable, and the
`--expert` decomposition (below) still isolates each knob for technical users.

### Attention Implementation

For LLM and SSM models, the attention/SDPA backend is an independent non-determinism source
(kernel selection can vary by shape). It is pinned and recorded, not left to HF defaults:

```python
model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="sdpa")
```

`attn_implementation` is embedded in every LLM/SSM result row.

### Benchmark Tiers

| Tier | Purpose | Model policy |
|------|---------|--------------|
| `quick` | Validate install and produce an immediate determinism/cost signal | Small, ungated models only; reduced run counts |
| `core` | Default representative suite | Covers CNN, vision transformer, decoder LLM, and recurrent/SSM-style inference where available |
| `extended` | Broader architecture and scale coverage | Adds gated, large, or specialized models such as Llama and CUDA Mamba kernels |

The default CLI run uses `core`. Optional models that are unavailable because of missing packages,
gated weights, or unsupported hardware are skipped with an explicit reason.

### Hardware Backends

| Backend | Runtime          | Determinism controls available |
|---------|------------------|-------------------------------|
| CUDA    | PyTorch + CUDA   | Full: cudnn flags, workspace config, use_deterministic_algorithms |
| MLX     | Apple MLX        | Limited: seed only (Metal is more deterministic by default) |
| CPU     | PyTorch CPU      | Fewer exposed controls; useful reference backend |

CPU is included as a reference / sanity-check backend, not as a performance target.
CPU results must record thread/library configuration because BLAS, oneDNN, OpenMP, and SIMD paths
can still affect numerical reproducibility.

---

## Determinism Presets (headline)

The headline story of this benchmark is a **two-point contrast**, not a four-level sweep. Most
people never touch the granular levels — they want to know one thing: *what does it cost to make
my model reproducible?* So the default axis is two named presets:

| Preset | Meaning | Knob bundle |
|--------|---------|-------------|
| `production` | What real serving runs: fast, non-reproducible | TF32 ON, cuDNN autotune ON, no deterministic algorithms, no workspace pin |
| `strict` | Bit-exact reproducibility, every control on | TF32 OFF, cuDNN autotune OFF + deterministic, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

`production` maps to the `NONE`-style baseline **plus** the fast production knobs (TF32, autotune).
`strict` maps to `FULL_DET`. The cost reported in the headline is the delta between these two —
the genuine, real-world tax someone pays to go from "fast default" to "bit-exact."

Bundling several knobs into one number is deliberate: a person complaining about non-determinism
does not care *which* knob cost them, only that "make it reproducible" cost (e.g.) 2×. The
`--expert` mode below decomposes that number for engineers who do care.

This is the default run:
```bash
python -m benchmark          # production vs strict, one delta per model
```

## Determinism Levels (expert decomposition)

Behind the two presets sit four granular levels, implemented as Python context managers in
`benchmark/determinism/modes.py`. Each level is a strict superset of the previous. These are the
`--expert` view — they isolate *where* the cost comes from (seed vs cuDNN vs cuBLAS).

```
NONE < SEED_ONLY < CUDNN_DET < FULL_DET
```

Note: `production` is **not** identical to `NONE`. `NONE` is bare PyTorch default; `production`
additionally turns on the fast inference knobs (TF32, cuDNN autotune) that real serving uses.
`strict` is identical to `FULL_DET`.

### NONE — Baseline

Default PyTorch/MLX behavior. All sources of non-determinism active:
- cuBLAS / cuDNN algorithm selection varies by run
- Parallel reductions (atomicAdd, warp scheduling) yield different FP orderings
- No seed fixed

**Why this is the baseline**: this is what users get out of the box. Everything is measured
relative to this.

### SEED_ONLY

Fix all PRNG seeds at run start:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
```

Eliminates: sampling noise (dropout if active, stochastic token sampling).
Does NOT eliminate: CUDA parallel reduction ordering, cuBLAS algorithm variation.

**Why include this level**: separates the cheap fix (seed) from the expensive fix (deterministic
algorithms). Most practitioners stop here and are surprised it doesn't give full reproducibility.

### CUDNN_DET

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`benchmark=False` is critical: cuDNN autotuner selects different convolution algorithms across
runs (and across input shapes). Each algorithm uses different reduction trees → different FP
results. Disabling it locks the algorithm, eliminating that source of variation.

`deterministic=True` forces cuDNN to use deterministic variants of selected algorithms where
available. Some algorithms have no deterministic variant and will error or fall back.

**Why a separate level**: this is the most common "make it deterministic" advice for vision
models. It has a real cost (no autotuning) that is worth isolating.

For models with no convolutions (LLM, SSM), `CUDNN_DET` is a no-op and collapses to `SEED_ONLY`;
the row's `effective_det_level` reflects this, so identical `SEED_ONLY`/`CUDNN_DET` rows are
expected for those classes rather than a bug.

### FULL_DET

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`CUBLAS_WORKSPACE_CONFIG`: forces cuBLAS to use deterministic GEMM algorithms. The workspace
config `:4096:8` selects a fixed workspace policy (commonly interpreted as 4096 KiB with 8
allocations); this prevents cuBLAS from selecting workspace-size-dependent algorithms that vary
run to run.

`use_deterministic_algorithms(True)`: PyTorch-level enforcement. Any op that has no
deterministic CUDA implementation raises `RuntimeError`. This is the strictest mode.

**Why this matters**: GEMMs (matrix multiplications) are the dominant op in all transformer and
CNN models. Making them deterministic is the biggest single cost.

On MLX: `FULL_DET` collapses to `SEED_ONLY`. Metal's execution model exposes fewer
non-determinism sources at the Python level.

### Requested vs Effective Determinism

Every result row records both:

| Field | Definition |
|-------|------------|
| `requested_det_level` | The mode requested by the user: `NONE`, `SEED_ONLY`, `CUDNN_DET`, or `FULL_DET` |
| `effective_det_level` | The strongest determinism mode actually enforceable on that backend/model |

This prevents backend no-ops from being misread. For example, MLX may report
`requested_det_level=FULL_DET` but `effective_det_level=SEED_ONLY` because MLX does not expose
CUDA-style deterministic algorithm controls.

### Determinism Result

The benchmark reports one headline reproducibility verdict per result row:

| Value | Meaning |
|-------|---------|
| `BIT_EXACT` | Every measured run produced identical raw output bits |
| `OUTPUT_STABLE` | Task-level output was identical, but raw tensors/logits differed |
| `DRIFTED` | Raw output or task-level output changed across runs |
| `UNSUPPORTED` | Requested determinism mode cannot be enforced on this backend/model |
| `ERROR` | Deterministic enforcement rejected an unsupported nondeterministic op |

`determinism_result` is empirical. `BIT_EXACT` means bit-identical across the configured repeated
runs on the recorded hardware/software stack. It is not a theoretical guarantee across machines,
drivers, library versions, or future executions.

### Determinism Achieved (safety column)

One dedicated enum answers the only question a safety-constrained user asks: **when `strict`
determinism was requested for this (model, batch), did the model produce identical output on every
run, on this fixed hardware?** It is evaluated on the `strict` run only and carries **no cost
information** — latency/memory penalties live in their own columns and are never folded in here.

| `determinism_achieved` | Meaning |
|------------------------|---------|
| `YES` | Under `strict`, every run produced bit-identical output |
| `NO`  | Under `strict`, output still varied across runs |
| `UNSUPPORTED` | `strict` controls cannot be enforced on this backend/model (e.g. MLX exposes no such controls, or a required op has no deterministic implementation) so it cannot be asserted |

This is empirical and scoped to the recorded stack (`gpu_name`, driver, `cuda_version`,
`cudnn_version`, `torch_version`, dtype) — a `YES` means "identical every run on *this* hardware,"
not a guarantee across different GPUs, drivers, or library versions.

---

## Metrics

### Universal (all model classes)

These are collected for every (model, backend, det_level) combination.

| Metric | Definition | Unit |
|--------|-----------|------|
| `latency_p50` | Median wall-clock inference time across N runs | ms |
| `latency_p95` | 95th percentile | ms |
| `latency_p99` | 99th percentile | ms |
| `det_overhead_pct` | `(det_p50 / baseline_p50 - 1) * 100` | % |
| `memory_peak_mb` | Peak inference memory (backend-specific source, see below) | MB |
| `memory_overhead_mb` | Delta vs NONE level | MB |
| `output_std` | Mean over tensor elements of per-element std across N runs (same input) | float |
| `output_max_abs_diff` | Max absolute difference of any run vs `run[0]` | float |
| `bit_exact_rate` | % of runs bit-identical to `run[0]` | % |
| `determinism_result` | Headline reproducibility verdict: `BIT_EXACT`, `OUTPUT_STABLE`, `DRIFTED`, `UNSUPPORTED`, or `ERROR` | enum |
| `warmup_latency_ms` | First-run latency (captures JIT / workspace alloc cost) | ms |

`output_std`, `output_max_abs_diff`, and `bit_exact_rate` measure **actual numerical
non-determinism** — they quantify how much the FP associativity problem matters in practice,
not just whether determinism mode is on.

**Reference run**: all numerical comparisons are made against `run[0]` (the first measurement
run), not all O(N²) run pairs. This is O(N), fully defined, and carries the same signal: if any
run differs from `run[0]`, the cell is not bit-exact. `output_std` reduces the per-element
standard deviation across the N runs to a single scalar by mean over tensor elements.

**`memory_peak_mb` source by backend**:

| Backend | Source | Notes |
|---------|--------|-------|
| CUDA | `torch.cuda.max_memory_allocated()` | Reset before each cell |
| MLX  | `mx.metal.get_peak_memory()` | Reset before each cell |
| CPU  | Process RSS delta (`psutil`) | Reported as a coarse reference; `null` if `psutil` unavailable |

**Baseline dependency**: `det_overhead_pct` and `memory_overhead_mb` are deltas against the
NONE row for the same (model, backend, dtype) cell. NONE is therefore **always executed** as the
baseline regardless of `--det-levels`; if NONE is explicitly excluded these two fields are emitted
as `null` rather than computed against a missing baseline.

### Vision-Specific

| Metric | Definition |
|--------|-----------|
| `throughput_img_s` | Images per second (batch_size / latency_p50) |

Input: fixed batch of synthetic ImageNet-sized tensors (224×224×3), batch size 32.

### LLM-Specific

| Metric | Definition |
|--------|-----------|
| `ttft_ms` | Time to first token: prompt submission → first output token (measures prefill) |
| `decode_tok_s` | Tokens per second during autoregressive decode |
| `token_repro_rate` | % of runs producing identical greedy-decoded token sequence |

`token_repro_rate` is measured at temperature=0 (greedy). Separate from `bit_exact_rate` on
logits: a model can produce different logit values but the same argmax token — both are measured.

Prefill input: fixed 512-token prompt. Decode: 128 tokens generated.

### SSM-Specific (extends LLM metrics)

SSMs have two distinct computation modes with different determinism properties:

| Metric | Definition | Availability |
|--------|-----------|--------------|
| `prefill_ttft_ms` | TTFT using parallel scan (training-mode forward) | CUDA + `mamba-ssm` only |
| `decode_tok_s_recurrent` | Tokens/sec using recurrent step (inference mode) | All backends |
| `prefill_repro_rate` | Bit-exact rate for prefill outputs | CUDA + `mamba-ssm` only |
| `decode_repro_rate` | Bit-exact rate for recurrent decode outputs | All backends |

**Why split prefill and decode for SSMs**: recurrent decode is inherently sequential and
deterministic even at `NONE` level. Prefill via parallel scan has the same FP ordering problem
as attention. Collapsing these would hide the key SSM property.

**Parallel-scan prefill is CUDA-only.** The pure-PyTorch fallback (MLX/CPU) implements only the
recurrent form — there is no parallel scan to measure — so `prefill_ttft_ms` and
`prefill_repro_rate` are emitted as `null` on those backends. Hypothesis 3's prefill-vs-decode
contrast can therefore only be validated on CUDA with the `mamba-ssm` kernel.

---

## Statistical Protocol

```
N_warmup = 10   # discarded, used to stabilize GPU clocks and fill caches
N_runs   = 50   # measurement runs
```

Rationale for N=50: gives stable p95/p99 estimates while keeping total benchmark time
tractable (<30 min on H100 for full suite). Configurable via CLI.

**Timing on CUDA**: use `torch.cuda.Event` (GPU-side timer), not Python wall clock.
`cuda.Event` records timestamps on the CUDA stream, eliminating Python interpreter jitter.

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
# ... inference ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

**Timing on MLX**: `time.perf_counter()` around `mx.eval(output)` + explicit sync.
MLX uses lazy evaluation; timing without `mx.eval()` measures graph construction, not execution.

**Timing on CPU**: `time.perf_counter()` is sufficient; no async execution.

**Input stability**: all runs within a benchmark cell use the identical input tensor (same seed,
pre-generated before warmup). This isolates determinism-level effects from input variation.

---

## Hardware Detection

On CUDA backend startup, auto-detect and record:

```python
{
  "gpu_name": torch.cuda.get_device_name(0),          # "NVIDIA H100 SXM5 80GB"
  "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
  "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
  "cuda_version": torch.version.cuda,
  "cudnn_version": torch.backends.cudnn.version(),
  "torch_version": torch.__version__,
}
```

On MLX:

```python
{
  "device": mx.default_device(),   # e.g. Device(gpu, 0)
  "mlx_version": mx.__version__,
  "metal": True,
}
```

This metadata is embedded in every result row so results from different machines are
unambiguously identifiable when aggregated.

CPU metadata also records thread and math-library settings when available, including
`torch.get_num_threads()`, `torch.get_num_interop_threads()`, oneDNN/MKL availability, and relevant
environment variables such as `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.

---

## Project Structure

```
ai-determinism/
├── SPEC.md                         ← this file
├── pyproject.toml
├── benchmark/
│   ├── __init__.py
│   ├── cli.py                      # entry point: python -m benchmark
│   ├── backends/
│   │   ├── base.py                 # abstract Backend interface
│   │   ├── cuda.py                 # CUDA backend (PyTorch)
│   │   ├── mlx.py                  # MLX backend (Apple Silicon)
│   │   └── cpu.py                  # CPU backend (reference / sanity check)
│   ├── determinism/
│   │   ├── modes.py                # context managers: NONE, SEED_ONLY, CUDNN_DET, FULL_DET
│   │   └── verify.py               # measure numerical reproducibility across runs
│   ├── models/
│   │   ├── vision.py               # ResNet-50, ViT-B/16 via HF
│   │   ├── llm.py                  # GPT-2, Llama-3.2-1B via HF
│   │   └── ssm.py                  # Mamba-370M; PyTorch fallback when mamba-ssm unavailable
│   ├── metrics/
│   │   ├── timing.py               # latency collection, warmup logic, cuda.Event wrapper
│   │   └── numerical.py            # std, max_abs_diff, bit_exact_rate
│   ├── runners/
│   │   ├── base_runner.py          # shared: warmup loop, measurement loop, metric assembly
│   │   ├── vision_runner.py
│   │   ├── llm_runner.py
│   │   └── ssm_runner.py
│   └── report/
│       ├── aggregator.py           # combine runs into summary stats
│       └── formatter.py            # table (rich), CSV, JSON output
└── results/                        # gitignored, output landing zone
```

---

## CLI Design

```bash
# THE default: production vs strict, headline cost per model
python -m benchmark

# Fast smoke test on current hardware
python -m benchmark --quick

# Expert decomposition: the full NONE/SEED_ONLY/CUDNN_DET/FULL_DET sweep
python -m benchmark --expert

# Representative tier selection
python -m benchmark --tier core
python -m benchmark --tier extended

# Restrict to model classes
python -m benchmark --models vision,llm

# Pick specific levels (implies --expert; production/strict are the default axis)
python -m benchmark --det-levels NONE,FULL_DET

# Specific backend (auto-detected by default)
python -m benchmark --backend mlx

# Precision (default: fp32 vision, bf16 LLM/SSM; this overrides uniformly)
python -m benchmark --dtype fp32

# Statistical parameters
python -m benchmark --n-runs 100 --n-warmup 20

# Output
python -m benchmark --output results/h100_run1.json --format table,json,csv
```

The primary axis is `production` vs `strict` (see §Determinism Presets). The granular four-level
sweep is opt-in via `--expert` (or implied when `--det-levels` is passed). This keeps the default
run a clean two-point cost story for the non-technical audience while preserving full diagnostic
depth for engineers.

Output formats:
- `table`: rich-formatted terminal table, color-coded overhead %
- `json`: machine-readable, includes full per-run latency arrays and hardware metadata
- `csv`: one row per (model, backend, det_level), for plotting

Default behavior should favor a successful first run:
- default run = `production` vs `strict` on the `core` tier, headline first
- `--quick` uses small, ungated models and reduced run counts
- `--tier core` is the default representative suite
- `--tier extended` enables gated, large, or CUDA-specialized models such as Llama and Mamba CUDA kernels
- Missing optional models are skipped with a clear reason instead of failing the whole benchmark

### Headline Output (for non-technical readers)

Before any table, the run prints a one-line-per-model verdict in plain language. The audience for
this benchmark is people who complain about non-determinism but will never read a metrics table,
so the headline must be a *sentence with a slowdown factor*, not a percentage:

```
Cost of bit-exact determinism (production → strict):

  Model                 Determinism   Slowdown      Throughput            Memory
  ResNet-50  (vision)   YES           2.1× slower   4,200 → 1,950 img/s   +16 MB
  ViT-B/16   (vision)   YES           1.7× slower   3,100 → 1,820 img/s   +12 MB
  GPT-2      (LLM)      YES           1.3× slower   first token +18 ms    +16 MB
  Llama-3.2-1B (LLM)    NO            —             —                     —
  Mamba-370M (SSM/MLX)  UNSUPPORTED   —             —                     —

  So what? Serving 1,000,000 images on ResNet-50:
    production: ~4.0 min   strict: ~8.5 min   →  +4.5 min for reproducibility
```

Design rules for the headline:
- **`Determinism` column is the first answer** — `YES` / `NO` / `UNSUPPORTED` from
  `determinism_achieved`. It says nothing about cost; cost lives in its own columns.
- **Slowdown factor (`2.1×`) is the lead cost number**, not overhead % — "twice as slow" is felt,
  "+110%" makes the reader do arithmetic. Cost columns are blank when `Determinism` ≠ `YES`
  (no point pricing reproducibility that wasn't achieved).
- **Vision and LLM rows first** — they carry the largest, most relatable cost.
- **A "so what" translation** converts the worst case into wall-clock (and optionally $) for a
  representative workload, so the cost is physical, not abstract.
- Models already bit-exact in `production` print `YES` with `no cost` — reproducibility was free.

Two new headline metrics back this section:

| Metric | Definition | Unit |
|--------|-----------|------|
| `slowdown_factor` | `strict_p50 / production_p50` | × |
| `throughput_drop` | production vs strict on the class throughput metric (img/s or tok/s) | pair |

### Primary Result Table

After the headline, a compact interpretation table for readers who want the breakdown:

| Model | Backend | Requested | Effective | Determinism | Latency Cost | Memory Cost | Verdict |
|-------|---------|-----------|-----------|-------------|--------------|-------------|---------|
| GPT-2 | CUDA | `FULL_DET` | `FULL_DET` | `BIT_EXACT` | `+7.8%` | `+16 MB` | Deterministic, moderate cost |
| GPT-2 | CUDA | `SEED_ONLY` | `SEED_ONLY` | `OUTPUT_STABLE` | `+0.2%` | `+0 MB` | Stable tokens, not bit-exact |
| ResNet-50 | CUDA | `NONE` | `NONE` | `DRIFTED` | baseline | baseline | Baseline is not reproducible |
| Llama-3.2-1B | MLX | `FULL_DET` | `SEED_ONLY` | `UNSUPPORTED` | `+0.0%` | `+0 MB` | Full controls unavailable |

#### Verdict generation rule

The `Verdict` string is derived deterministically from `determinism_result` and the latency cost
band, so the headline column is reproducible rather than hand-written.

Cost bands (on `det_overhead_pct`): `low` < 2%, `moderate` 2–10%, `high` > 10%.

| `determinism_result` | Verdict template |
|----------------------|------------------|
| `BIT_EXACT` | `Deterministic, {band} cost` |
| `OUTPUT_STABLE` | `Stable tokens, not bit-exact` |
| `DRIFTED` (NONE level) | `Baseline is not reproducible` |
| `DRIFTED` (det level) | `Determinism not achieved despite controls` |
| `UNSUPPORTED` | `Full controls unavailable` |
| `ERROR` | `Nondeterministic op rejected under FULL_DET` |

---

## Example Output

Illustrative output of the default run on an H100 (numbers are representative, not measured).

### Default run

```console
$ python -m benchmark

AI Determinism Benchmark — measuring the cost of reproducibility
Hardware: NVIDIA H100 SXM5 80GB · CUDA 12.4 · cuDNN 9.1 · torch 2.5.1
Tier: core · dtype: fp32 (vision) / bf16 (llm,ssm) · N=50 runs, 10 warmup
Comparing: production (fast)  vs  strict (bit-exact)

  [1/4] ResNet-50    vision  ████████████████████ production ✓  strict ✓
  [2/4] ViT-B/16     vision  ████████████████████ production ✓  strict ✓
  [3/4] GPT-2        llm     ████████████████████ production ✓  strict ✓
  [4/4] Mamba-370M   ssm     ████████████████████ production ✓  strict ✓   (mamba-pytorch fallback)

  skipped: Llama-3.2-1B (gated — needs HF token; use --tier extended)
```

```
═══════════════════════════════════════════════════════════════════════════════
  Can determinism be reached? And what does it cost? (production → strict)
═══════════════════════════════════════════════════════════════════════════════

  Model                 Determinism   Slowdown      Throughput            Memory
  ─────────────────────────────────────────────────────────────────────────────
  ResNet-50  (vision)   YES           2.1× slower   4,200 → 1,950 img/s   +16 MB
  ViT-B/16   (vision)   YES           1.6× slower   3,100 → 1,920 img/s   +12 MB
  GPT-2      (LLM)      YES           1.3× slower   first token +18 ms    +16 MB
  Mamba-370M (SSM)      YES           1.0× (free)   decode unchanged      +0 MB

  So what? Serving 1,000,000 images on ResNet-50:
    production ≈ 4.0 min   strict ≈ 8.5 min   →  +4.5 min for reproducibility
═══════════════════════════════════════════════════════════════════════════════
```

The breakdown table follows the headline:

```
  Primary results
  ───────────────────────────────────────────────────────────────────────────────────
  Model       Backend  Requested  Effective  Determinism  Latency Cost  Mem Cost  Verdict
  ───────────────────────────────────────────────────────────────────────────────────
  ResNet-50   CUDA     strict     FULL_DET   YES          +110%         +16 MB    Deterministic, high cost
  ViT-B/16    CUDA     strict     FULL_DET   YES          +63%          +12 MB    Deterministic, high cost
  GPT-2       CUDA     strict     FULL_DET   YES          +28%          +16 MB    Deterministic, high cost
  Mamba-370M  CUDA     strict     FULL_DET   YES          +0.4%         +0 MB     Deterministic, low cost
  ───────────────────────────────────────────────────────────────────────────────────
  Determinism: YES = bit-identical every run on this hardware · NO = still varied · UNSUPPORTED = controls unavailable
  Scope: empirical for this exact GPU + driver + library stack only.

  Results written: results/h100_run1.json · results/h100_run1.csv
```

### When determinism cannot be reached

The `Determinism` column states plainly when bit-exactness is off the table. On MLX (no
deterministic-algorithm controls):

```
  Model                 Determinism   Slowdown      Throughput            Memory
  ─────────────────────────────────────────────────────────────────────────────
  ResNet-50  (vision)   UNSUPPORTED   —             —                     —
  GPT-2      (LLM)      UNSUPPORTED   —             —                     —
  Mamba-370M (SSM)      YES           1.0× (free)   decode unchanged      +0 MB

  Note: MLX exposes no deterministic-algorithm controls (effective level = SEED_ONLY).
        Bit-exactness cannot be asserted on this backend.
```

A hard negative — an op with no deterministic kernel under `strict`:

```
  SomeModel  (vision)   NO            —             —                     —
  Note: strict raised RuntimeError — 'upsample_bilinear2d_backward' has no deterministic
        implementation. Determinism cannot be reached for this model on CUDA.
```

### Expert decomposition

```console
$ python -m benchmark --expert --models vision
```
```
  Model       Level       Determinism  bit_exact_rate  p50      vs NONE
  ─────────────────────────────────────────────────────────────────────
  ResNet-50   NONE        NO           0%              7.6 ms   baseline
  ResNet-50   SEED_ONLY   NO           0%              7.6 ms   +0.1%
  ResNet-50   CUDNN_DET   NO           34%             9.9 ms   +30%      ← autotune off
  ResNet-50   FULL_DET    YES          100%            16.0 ms  +110%     ← cuBLAS deterministic GEMM
```

Reading: the jump to `FULL_DET` (deterministic cuBLAS GEMM) is where both bit-exactness and the
bulk of the cost appear — the "where does the cost come from" view.

### Machine-readable output

`results/h100_run1.csv` — one row per model × backend × level:

```csv
model,backend,requested_det_level,effective_det_level,dtype,determinism_achieved,latency_p50_ms,det_overhead_pct,slowdown_factor,memory_overhead_mb,bit_exact_rate,gpu_name,cuda_version,cudnn_version,torch_version,n_runs
ResNet-50,CUDA,production,production,fp32,n/a,7.6,0.0,1.0,0,0,NVIDIA H100 SXM5 80GB,12.4,9.1,2.5.1,50
ResNet-50,CUDA,strict,FULL_DET,fp32,YES,16.0,110.5,2.11,16,100,NVIDIA H100 SXM5 80GB,12.4,9.1,2.5.1,50
```

`results/h100_run1.json` — excerpt (full file includes per-run latency arrays):

```json
{
  "hardware": {
    "gpu_name": "NVIDIA H100 SXM5 80GB", "cuda_version": "12.4",
    "cudnn_version": 91002, "torch_version": "2.5.1", "tf32_enabled_production": true
  },
  "results": [
    {
      "model": "ResNet-50", "backend": "CUDA", "dtype": "fp32",
      "requested_det_level": "strict", "effective_det_level": "FULL_DET",
      "determinism_achieved": "YES",
      "bit_exact_rate": 100.0, "output_max_abs_diff": 0.0,
      "latency_p50_ms": 16.0, "latency_p95_ms": 16.4,
      "slowdown_factor": 2.11, "det_overhead_pct": 110.5, "memory_overhead_mb": 16,
      "scope": "Empirical for this exact GPU/driver/library stack only."
    }
  ]
}
```

---

## Mamba / SSM Implementation Notes

`mamba-ssm` is a CUDA-only package (custom CUDA kernels). On MLX and CPU:

- Fall back to a pure-PyTorch reference implementation of the Mamba recurrence
- This fallback is slower but functionally correct and backend-portable
- The fallback does NOT use the parallel scan — it implements only the recurrent form
- Label results clearly: `mamba-ssm (cuda kernel)` vs `mamba-pytorch (fallback)`

The pure-PyTorch fallback is also useful as a determinism reference because its recurrent path is
sequential and backend-portable. CPU fallback results are treated as an empirical reference, not a
theoretical ground truth.

---

## Expected Findings (Hypotheses to Validate)

These are predictions, not assumptions — the benchmark exists to confirm or refute them:

1. **Vision / CUDNN_DET** will show the largest single overhead because cuDNN autotuning is
   a significant optimization for convolutions.

2. **LLM decode** will show smaller overhead than prefill at FULL_DET, because decode is
   dominated by memory bandwidth (KV cache reads), not compute.

3. **Mamba decode (recurrent)** will show ~0% overhead at all det levels — sequential
   recurrence is deterministic by construction, so determinism modes add nothing.

4. **MLX** will show near-zero overhead for all det levels (no knobs → no cost to turning them on).

5. **`FULL_DET` memory overhead** may be measurable due to
   `CUBLAS_WORKSPACE_CONFIG` allocating fixed workspace buffers.

---

## Non-Goals

- We do not measure training — only inference
- We do not measure accuracy degradation from quantization or approximation
- We do not benchmark across different batch sizes (fixed per model class)
- We do not profile kernel-level with Nsight / Instruments — only Python-level timing
- We do not test ROCm / XPU — CUDA + MLX covers the primary targets
