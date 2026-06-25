"""Decoder LLM — GPT-2 (SPEC §Scope). Two backend-native paths:

  * torch + HuggingFace transformers (CPU here, CUDA later), attn_implementation=sdpa
  * Apple MLX-native via mlx_lm (the headline local path on this Mac)

The timed op is prefill (a forward over the fixed prompt -> logits). Decode metrics
(ttft, decode tok/s, token reproducibility) are measured separately over a few
greedy generations.
"""

from __future__ import annotations

import numpy as np

from benchmark.models import ModelUnavailable

GPT2_REPO = "gpt2"
GPT2_VOCAB = 50257

_TORCH_DTYPE = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def _prompt_ids_np(seed: int, prompt_len: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, GPT2_VOCAB, size=(1, prompt_len), dtype=np.int64)


class GPT2TorchCell:
    model_class = "llm"
    display_name = "GPT-2"

    def __init__(self, backend, dtype: str, cfg):
        import torch
        from transformers import AutoModelForCausalLM

        self.backend = backend
        self.cfg = cfg
        self.dtype_str = dtype
        tdtype = getattr(torch, _TORCH_DTYPE[dtype])
        self.attn_implementation = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(
            GPT2_REPO, attn_implementation="sdpa", dtype=tdtype
        ).eval()

    def make_input(self):
        import torch

        return torch.from_numpy(_prompt_ids_np(self.cfg.seed, self.cfg.prompt_len)).long()

    def infer(self, ids):
        import torch

        with torch.no_grad():
            return self.model(ids).logits  # [1, prompt_len, vocab]

    def decode_greedy(self, ids, n_tokens: int) -> list[int]:
        import torch

        tokens: list[int] = []
        with torch.no_grad():
            out = self.model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(int(nxt))
            for _ in range(n_tokens - 1):
                out = self.model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
                tokens.append(int(nxt))
        return tokens

    def extra_metadata(self) -> dict:
        return {"attn_implementation": self.attn_implementation}


class GPT2MLXCell:
    model_class = "llm"
    display_name = "GPT-2"

    def __init__(self, backend, dtype: str, cfg):
        from mlx_lm import load

        self.backend = backend
        self.cfg = cfg
        self.dtype_str = dtype
        self.attn_implementation = "mlx-native"
        self.model, self.tokenizer = load(GPT2_REPO)

    def make_input(self):
        import mlx.core as mx

        return mx.array(_prompt_ids_np(self.cfg.seed, self.cfg.prompt_len))

    def infer(self, ids):
        return self.model(ids)  # [1, prompt_len, vocab]

    def decode_greedy(self, ids, n_tokens: int) -> list[int]:
        import mlx.core as mx

        cur = ids
        for _ in range(n_tokens):
            logits = self.model(cur)[:, -1, :]
            nxt = mx.argmax(logits, axis=-1)[:, None]
            cur = mx.concatenate([cur, nxt], axis=1)
        mx.eval(cur)
        return [int(t) for t in cur[0, ids.shape[1]:].tolist()]

    def extra_metadata(self) -> dict:
        return {"attn_implementation": self.attn_implementation}


def load_gpt2(backend, dtype: str, cfg):
    if backend.NAME == "mlx":
        return GPT2MLXCell(backend, dtype, cfg)
    if backend.NAME == "cuda":  # pragma: no cover - stub backend
        raise ModelUnavailable("CUDA backend is a stub on this host")
    return GPT2TorchCell(backend, dtype, cfg)
