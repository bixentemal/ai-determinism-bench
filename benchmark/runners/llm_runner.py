"""LLM-specific measurement (SPEC §LLM-Specific): time-to-first-token, decode
throughput, and greedy token reproducibility. Run under the active determinism
context so the decode path sees the same knobs as prefill."""

from __future__ import annotations

import statistics
import time


def measure(cell, backend, inp, cfg):
    seqs: list[list[int]] = []
    decode_times: list[float] = []
    for _ in range(cfg.decode_repeats):
        t0 = time.perf_counter()
        toks = cell.decode_greedy(inp, cfg.decode_tokens)
        decode_times.append(time.perf_counter() - t0)
        seqs.append(toks)

    decode_tok_s = cfg.decode_tokens / statistics.median(decode_times)
    token_repro_rate = 100.0 * sum(s == seqs[0] for s in seqs) / len(seqs)

    # TTFT proxy: a single prefill forward, device-synced timing.
    ttft_ms, _ = backend.time_call(lambda: cell.infer(inp))

    extra = {
        "ttft_ms": ttft_ms,
        "decode_tok_s": decode_tok_s,
        "token_repro_rate": token_repro_rate,
    }
    task_stable = token_repro_rate >= 100.0
    return extra, task_stable
