"""Vision models — ResNet-50 and ViT-B/16 (SPEC §Scope), FP32 by default.

Two backend-native paths, mirroring the LLM module:
  * torch + torchvision (CPU here, CUDA later)
  * Apple MLX-native via mlx-image (`mlxim`), pretrained weights from the HF Hub

ResNet and ViT run the identical task on the same backend, which is the SPEC's
clearest comparison: cost of making convolution deterministic vs attention.
"""

from __future__ import annotations

import numpy as np

from benchmark.models import ModelUnavailable

_TORCH_DTYPE = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}

# key -> (torchvision weights-enum attr, enum member, builder fn)
_VISION_MODELS = {
    "resnet50": ("ResNet50_Weights", "IMAGENET1K_V2", "resnet50"),
    "vit_b_16": ("ViT_B_16_Weights", "IMAGENET1K_V1", "vit_b_16"),
}


class VisionTorchCell:
    model_class = "vision"

    def __init__(self, backend, dtype: str, cfg, display_name: str, key: str):
        import torch
        import torchvision

        self.backend = backend
        self.cfg = cfg
        self.display_name = display_name
        self.dtype_str = dtype
        self._tdtype = getattr(torch, _TORCH_DTYPE[dtype])

        enum_attr, member, fn_name = _VISION_MODELS[key]
        weights = getattr(getattr(torchvision.models, enum_attr), member)
        builder = getattr(torchvision.models, fn_name)
        self.model = builder(weights=weights).eval().to(self._tdtype)

    def make_input(self):
        import torch

        g = torch.Generator().manual_seed(self.cfg.seed)
        return torch.randn(
            self.cfg.batch_size, 3, self.cfg.image_size, self.cfg.image_size,
            generator=g, dtype=torch.float32,
        ).to(self._tdtype)

    def infer(self, x):
        import torch

        with torch.no_grad():
            return self.model(x)  # logits [batch, 1000]

    def task_signature(self, output_np):
        return output_np.argmax(axis=-1)  # predicted class ids

    def extra_metadata(self) -> dict:
        return {"attn_implementation": None, "impl": "torchvision"}


class VisionMLXCell:
    model_class = "vision"

    # MLX-native builders, keyed like the registry.
    _BUILDERS = {
        "resnet50": "benchmark.models._mlx_resnet",
        "vit_b_16": "benchmark.models._mlx_vit",
    }

    def __init__(self, backend, dtype: str, cfg, display_name: str, key: str):
        import importlib

        self.backend = backend
        self.cfg = cfg
        self.display_name = display_name
        self.dtype_str = "fp32"  # MLX-native weights are FP32
        module = importlib.import_module(self._BUILDERS[key])
        self.model = module.load_pretrained()

    def make_input(self):
        import mlx.core as mx

        rng = np.random.default_rng(self.cfg.seed)
        # MLX vision is NHWC.
        arr = rng.standard_normal(
            (self.cfg.batch_size, self.cfg.image_size, self.cfg.image_size, 3)
        ).astype(np.float32)
        return mx.array(arr)

    def infer(self, x):
        return self.model(x)  # logits [batch, 1000]

    def task_signature(self, output_np):
        return output_np.argmax(axis=-1)

    def extra_metadata(self) -> dict:
        return {"attn_implementation": None, "impl": "mlx-image"}


def _load(backend, dtype, cfg, display_name, key):
    if backend.NAME == "mlx":
        return VisionMLXCell(backend, dtype, cfg, display_name, key)
    if backend.NAME == "cuda":  # pragma: no cover - stub backend
        raise ModelUnavailable("CUDA backend is a stub on this host")
    return VisionTorchCell(backend, dtype, cfg, display_name, key)


def load_resnet50(backend, dtype: str, cfg):
    return _load(backend, dtype, cfg, "ResNet-50", "resnet50")


def load_vit(backend, dtype: str, cfg):
    return _load(backend, dtype, cfg, "ViT-B/16", "vit_b_16")
