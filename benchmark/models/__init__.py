"""Model loaders. Each loader returns a model "cell" or raises ModelUnavailable
with a human-readable reason, which the runner surfaces as an explicit skip
(SPEC: missing/unsupported models are skipped with a reason, not a hard failure).
"""

from __future__ import annotations

from dataclasses import dataclass


class ModelUnavailable(Exception):
    """Raised when a (model, backend) combination cannot run; carries the reason."""


@dataclass
class ModelSpec:
    key: str
    display_name: str
    model_class: str  # "vision" | "llm"
    loader: str       # module:function


# Registry of models available in this slice. Llama / Mamba are fast-follow.
REGISTRY: dict[str, ModelSpec] = {
    "resnet50": ModelSpec("resnet50", "ResNet-50", "vision", "benchmark.models.vision:load_resnet50"),
    "vit": ModelSpec("vit", "ViT-B/16", "vision", "benchmark.models.vision:load_vit"),
    "gpt2": ModelSpec("gpt2", "GPT-2", "llm", "benchmark.models.llm:load_gpt2"),
}

# Class -> model keys, for --models filtering.
CLASS_MODELS: dict[str, list[str]] = {
    "vision": ["resnet50", "vit"],
    "llm": ["gpt2"],
}

# Tier -> ordered model keys (slice covers quick == core).
TIER_MODELS: dict[str, list[str]] = {
    "quick": ["resnet50", "vit", "gpt2"],
    "core": ["resnet50", "vit", "gpt2"],
}


def select_models(tier: str, classes: list[str] | None) -> list[ModelSpec]:
    keys = TIER_MODELS[tier]
    if classes:
        allowed = {k for c in classes for k in CLASS_MODELS.get(c, [])}
        keys = [k for k in keys if k in allowed]
    return [REGISTRY[k] for k in keys]


def load_cell(spec: ModelSpec, backend, dtype: str, cfg):
    module_name, fn_name = spec.loader.split(":")
    import importlib

    fn = getattr(importlib.import_module(module_name), fn_name)
    return fn(backend, dtype, cfg)
