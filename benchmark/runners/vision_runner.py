"""Vision-specific measurement (SPEC §Vision-Specific). Throughput is derived from
p50 in the aggregator; task stability (argmax class ids) is tracked in the base loop,
so there is no extra per-run work here."""

from __future__ import annotations


def measure(cell, backend, inp, cfg):
    # No extra timed work; throughput_img_s is computed from latency_p50 downstream.
    return {}, None
