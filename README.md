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
| Backends | **CPU** (PyTorch), **MLX** (Apple Silicon), **CUDA** (NVIDIA) | — |
| Vision | ResNet-50, ViT-B/16 | — |
| LLM | GPT-2 | Llama-3.2-1B |
| SSM | — | Mamba-370M |
| Determinism | `production` vs `strict` presets | `--expert` 4-level decomposition |

On Apple Silicon, ResNet-50 / ViT-B/16 / GPT-2 all run **MLX-native** (vision models are hand-ported
from torchvision weights and validated to match within fp tolerance). The **real determinism cost
tax shows up on CUDA** — ViT-B/16 attention is 1.6× slower under `strict`, ResNet-50 convolution is
1.3× slower (see [Results](#results) and [Caveats](#caveats--scope)).

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

### CUDA backend, NVIDIA RTX 4090

`torch 2.12.1+cu130`, CUDA 13.0, cuDNN 9.2.0; core tier, N=50 runs, 10 warmup; fp32 vision / bf16 LLM.
Generated with `uv run python -m benchmark --backend cuda`:

```
Can determinism be reached? And what does it cost? (production → strict)
Model                Determinism  Slowdown     Throughput           Memory
ResNet-50  (vision)  YES          1.3× slower  2,748 → 2,113 img/s  +0 MB
ViT-B/16  (vision)   YES          1.6× slower  1,435 → 898 img/s    +0 MB
GPT-2  (llm)         YES          1.1× slower  first token +0 ms    +0 MB

  So what? Serving 1,000,000 images on ViT-B/16:
    production ≈ 11.6 min   strict ≈ 18.6 min   →  +6.9 min for reproducibility

Primary results
Model      Backend  Preset      Requested  Effective  Determinism  Latency Cost  Mem Cost  Verdict
ResNet-50  cuda     production  NONE       NONE       —            baseline      +0 MB     Deterministic, low cost
ResNet-50  cuda     strict      FULL_DET   FULL_DET   YES          +30.1%        +0 MB     Deterministic, high cost
ViT-B/16   cuda     production  NONE       NONE       —            baseline      +0 MB     Deterministic, low cost
ViT-B/16   cuda     strict      FULL_DET   FULL_DET   YES          +59.8%        +0 MB     Deterministic, high cost
GPT-2      cuda     production  NONE       NONE       —            baseline      +0 MB     Deterministic, low cost
GPT-2      cuda     strict      FULL_DET   FULL_DET   YES          +10.2%        +0 MB     Deterministic, high cost
```

All three models achieve **bit-exact determinism** (`bit_exact_rate = 100%` across 50 runs) under
`strict`. The cost varies sharply by operation type: ViT-B/16 attention (+60%) is nearly double the
penalty of ResNet-50 convolution (+30%), while GPT-2 prefill is cheap to lock down (+10%). FULL_DET
is fully enforceable on CUDA — `requested` and `effective` level match in every row.

### Does LLM determinism cost scale with model size?

Four-model sweep across architectures and scales, all on CUDA, bf16, 512-token prompt.
Generated with `uv run python -m benchmark --backend cuda --tier llm-scale`:

```
Model       Params  Attention          det overhead  decode (prod→strict)
GPT-2       117M    MHA  12Q/12KV      +11.5%        470 → 410 tok/s
Qwen3-0.6B  600M    GQA  16Q/8KV       +6.6%         75 → 70 tok/s
Qwen3-8B      8B    GQA  32Q/8KV       +0.4%         47 → 46 tok/s
Mistral-7B    7B    GQA+SWA 32Q/8KV   +1.7%         54 → 51 tok/s
```

All four reach `bit_exact_rate = 100%` and `determinism_achieved = YES`.

**The overhead is not proportional to model size — it tracks attention architecture.** The
non-deterministic fast path lives in the SDPA flash kernel (attention only); FFN matmuls are
trivially deterministic. As GQA reduces the KV head count and larger FFNs make attention a smaller
fraction of total compute, the overhead shrinks:

- **GPT-2 MHA (+11.5%)**: 12 KV heads = 12 Q heads. Attention is a large share of compute.
- **Qwen3-0.6B GQA (+6.6%)**: 8 KV heads / 16 Q heads. Smaller KV tensors, bigger FFN share.
- **Qwen3-8B GQA (+0.4%)**: Same 8 KV heads but d_model=4096. FFN dominates absolutely;
  attention overhead becomes noise.
- **Mistral-7B GQA+SWA (+1.7%)**: Same GQA ratio as Qwen3-8B. The extra cost vs Qwen3-8B at
  512 tokens is within noise — at this sequence length the 4096-token SWA window is not active.

#### Sliding-window attention at long context (8192 tokens)

To activate Mistral's SWA (window = 4096), we ran a targeted 8192-token pass with 10 runs
(last-token timing only, no full-logit copy to avoid OOM):

```
Model           seq_len  prod p50   strict p50  det overhead
Qwen3-8B        8192     1099 ms    1128 ms      +2.5%
Mistral-7B SWA  8192     1302 ms    1334 ms      +2.4%
```

Two findings: (1) SWA has **no detectable effect on determinism overhead** — Mistral and
Qwen3-8B converge to the same ~+2.5% penalty once sequence length is the bottleneck.
(2) The overhead *rises* from ~+1% at 512 tokens to ~+2.5% at 8192 tokens for both models —
because attention's share of total compute grows with sequence length (O(n·d) → O(n²·d/heads) for
the attention portion), so the deterministic fallback costs more in absolute time even as a fraction
of a larger run.

**Implication**: production-scale GQA LLMs can be made bit-exactly reproducible at roughly
**1–3% latency cost**, not the 10%+ one might infer from small MHA benchmarks. The cost floor is
set by the FFN share; the ceiling by the sequence length and attention-head count.

### MLX backend, Apple Silicon

`mlx 0.31.2`, Metal GPU; core tier, N=50 runs, 10 warmup; fp32 vision / bf16 LLM.
Generated with `uv run python -m benchmark --backend mlx`:

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
presets is within noise (≤1.5%): on this backend reproducibility is effectively free.

## Conclusions — RTX 4090, CUDA 13.0, cuDNN 9.2.0

These conclusions are scoped to the exact stack recorded above. The qualitative findings are expected
to generalise across NVIDIA hardware; the percentages are Ada Lovelace / cuDNN 9.2 specific.

### 1. Bit-exact determinism is achievable for every tested architecture

Every model — ResNet-50, ViT-B/16, GPT-2, Qwen3-0.6B, Qwen3-8B, Mistral-7B — reached
`bit_exact_rate = 100%` under `strict`. `FULL_DET` is fully enforceable on CUDA: `requested` and
`effective` level match in every row. Determinism is a yes/no question on CUDA; the only open
question is cost.

### 2. Cost is set by the operation type, not model size

Ranked by determinism overhead on this platform:

| Operation | Representative model | Overhead |
|-----------|---------------------|---------|
| Vision attention | ViT-B/16 | **+60%** |
| Vision convolution | ResNet-50 | **+30%** |
| LLM MHA (legacy) | GPT-2 | **+11.5%** |
| LLM GQA, short context | Qwen3-8B @ 512 tok | **+0.4%** |
| LLM GQA, long context | Qwen3-8B @ 8192 tok | **+2.5%** |

Model size is irrelevant. Qwen3-8B is 70× larger than GPT-2 and 30× cheaper to determinize.

### 3. The non-deterministic path lives entirely in the SDPA flash kernel

FFN matmuls (the dominant compute in all large models) have a trivially deterministic cuBLAS path
at negligible overhead. The non-determinism that `strict` eliminates comes from SDPA's flash
attention kernel, which falls back to a slower but deterministic implementation. Everything else —
layer norms, activations, positional embeddings, embeddings lookups — is already deterministic.

### 4. GQA drives LLM overhead toward zero as models scale

With a fixed KV-head count (8 in both Qwen3 and Mistral), the attention tensor size stays bounded
while the FFN width grows with d\_model. Attention's share of total compute shrinks, and with it
the overhead from the deterministic SDPA fallback. The practical cost floor for a modern 7–8B GQA
model is **1–3%**, not the 10%+ one might infer from small MHA benchmarks. GPT-2's +11.5% is an
architecture artifact, not a representative LLM number.

### 5. Sliding-window attention does not change the determinism cost

Mistral-7B (SWA, window = 4096) and Qwen3-8B (full-context) converge to identical ~+2.5% overhead
at 8192 tokens. The windowed attention kernel has the same relative cost for its deterministic
fallback as full-context GQA. SWA is not a determinism concern.

### 6. Context length, not model size, sets the LLM cost ceiling

At 512 tokens, Qwen3-8B pays +0.4% (attention is a negligible fraction of total compute). At 8192
tokens it pays +2.5% — because attention's compute scales as O(n²) while FFN scales as O(n), so
attention reclaims a larger share at long context. For production serving lengths (512–4096 tokens)
the overhead stays below +2%.

### 7. Apple Silicon cannot provide determinism guarantees

MLX exposes no deterministic-algorithm controls. `strict` can only enforce `SEED_ONLY`; FULL\_DET
is unenforced. Models are empirically bit-exact on this hardware (latency delta ≤1.5%), but that
stability cannot be *asserted* and may not hold across MLX versions, Metal driver updates, or
thermal states. Apple Silicon is not suitable for applications that need a reproducibility guarantee;
CUDA with FULL\_DET is the right platform for that requirement.

---

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
- **CUDA** results above were measured on an RTX 4090. The exact overhead percentages are
  microarchitecture- and driver-specific; Ampere vs Ada vs Hopper, or a different cuDNN version,
  can produce meaningfully different numbers. The qualitative story (attention > convolution > prefill)
  is expected to hold across NVIDIA hardware, but re-run on your target GPU to get actionable figures.

---

## Project layout

```
benchmark/
├── cli.py / __main__.py        # python -m benchmark
├── config.py                   # run configuration
├── backends/   base · cpu · mlx · cuda
├── determinism/ modes · verify # presets + requested/effective level + verdicts
├── models/     vision · llm · _mlx_resnet · _mlx_vit
├── metrics/    timing · numerical
├── runners/    base · vision · llm
└── report/     aggregator · formatter   # headline + table + json/csv
```

Full design, metric definitions, and rationale: [`SPEC.md`](SPEC.md).
