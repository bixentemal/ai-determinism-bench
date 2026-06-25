"""Decoder LLMs. Two backend-native paths:

  * torch + HuggingFace transformers (CPU / CUDA)
  * Apple MLX-native via mlx_lm (GPT-2 only)

The timed op is prefill (a forward over the fixed prompt -> logits). Decode metrics
(ttft, decode tok/s, token reproducibility) are measured separately over a few
greedy generations.
"""

from __future__ import annotations

import numpy as np

GPT2_REPO = "gpt2"
GPT2_VOCAB = 50257

_TORCH_DTYPE = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def _prompt_ids_np(seed: int, prompt_len: int, vocab_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(1, prompt_len), dtype=np.int64)


class TorchCausalLMCell:
    """Generic HuggingFace CausalLM cell for any decoder model."""

    model_class = "llm"

    def __init__(self, backend, dtype: str, cfg, repo: str, display_name: str, attn_impl: str | None = "sdpa"):
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig

        self.backend = backend
        self.cfg = cfg
        self.dtype_str = dtype
        self.display_name = display_name
        tdtype = getattr(torch, _TORCH_DTYPE[dtype])

        # Resolve vocab size from config so prompt IDs stay in-range.
        model_cfg = AutoConfig.from_pretrained(repo)
        self.vocab_size = model_cfg.vocab_size

        kwargs = {"dtype": tdtype}
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl
        self.attn_implementation = attn_impl or "eager"
        self.model = AutoModelForCausalLM.from_pretrained(repo, **kwargs).eval().to(backend.device)

    def make_input(self):
        import torch

        ids = _prompt_ids_np(self.cfg.seed, self.cfg.prompt_len, self.vocab_size)
        return torch.from_numpy(ids).long().to(self.backend.device)

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

        return mx.array(_prompt_ids_np(self.cfg.seed, self.cfg.prompt_len, GPT2_VOCAB))

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
    return TorchCausalLMCell(backend, dtype, cfg, repo=GPT2_REPO, display_name="GPT-2", attn_impl="sdpa")


def load_qwen3_0b6(backend, dtype: str, cfg):
    return TorchCausalLMCell(backend, dtype, cfg, repo="Qwen/Qwen3-0.6B", display_name="Qwen3-0.6B", attn_impl="sdpa")


def load_qwen3_8b(backend, dtype: str, cfg):
    # GQA 32Q/8KV, full-context attention, d_model=4096
    return TorchCausalLMCell(backend, dtype, cfg, repo="Qwen/Qwen3-8B", display_name="Qwen3-8B", attn_impl="sdpa")


def load_mistral_7b(backend, dtype: str, cfg):
    # GQA 32Q/8KV + sliding-window attention (window=4096). At prompt_len<=4096 the
    # SWA kernel is identical to full attention; use --prompt-len 8192 to activate it.
    return TorchCausalLMCell(backend, dtype, cfg, repo="mistralai/Mistral-7B-v0.1", display_name="Mistral-7B", attn_impl="sdpa")
