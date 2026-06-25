"""MLX-native ViT-B/16, weights ported from torchvision's vit_b_16.

torchvision packs Q/K/V into a single `in_proj_weight` (2304×768) inside
`nn.MultiheadAttention`; we keep that raw layout and do the attention math
explicitly so the pretrained weights map over without splitting. Patch ordering
(H-major) and the prepended class token + positional embedding match torchvision,
so logits agree to fp tolerance.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)  # torchvision mlp.0
        self.fc2 = nn.Linear(hidden, dim)  # torchvision mlp.3

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        # Raw packed projection (matches torchvision MultiheadAttention).
        self.in_proj_weight = mx.zeros((3 * dim, dim))
        self.in_proj_bias = mx.zeros((3 * dim,))
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x):
        B, N, D = x.shape
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias  # [B, N, 3D]
        q, k, v = mx.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = Attention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden)

    def __call__(self, x):
        x = x + self.self_attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViTB16(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, hidden=3072, patch=16, img=224, classes=1000):
        super().__init__()
        self.conv_proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)  # NHWC
        self.class_token = mx.zeros((1, 1, dim))
        n = (img // patch) ** 2
        self.pos_embedding = mx.zeros((1, n + 1, dim))
        self.layers = [Block(dim, heads, hidden) for _ in range(depth)]
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def __call__(self, x):  # x: NHWC [B, 224, 224, 3]
        x = self.conv_proj(x)            # [B, 14, 14, 768]
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)       # H-major patch flatten
        cls = mx.broadcast_to(self.class_token, (B, 1, D))
        x = mx.concatenate([cls, x], axis=1) + self.pos_embedding
        for blk in self.layers:
            x = blk(x)
        x = self.ln(x)
        return self.head(x[:, 0])        # class-token logits


def _translate_key(k: str) -> str | None:
    """torchvision vit_b_16 state_dict key -> this module's flattened key."""
    if k.endswith("num_batches_tracked"):
        return None
    k = k.replace("encoder.layers.encoder_layer_", "layers.")
    k = k.replace("encoder.pos_embedding", "pos_embedding")
    k = k.replace("encoder.ln.", "ln.")
    k = k.replace("heads.head.", "head.")
    k = k.replace(".mlp.0.", ".mlp.fc1.")
    k = k.replace(".mlp.3.", ".mlp.fc2.")
    return k


def load_pretrained() -> ViTB16:
    import torchvision

    model = ViTB16()
    model.eval()
    tv = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    ).eval()

    weights = []
    for key, tensor in tv.state_dict().items():
        mlx_key = _translate_key(key)
        if mlx_key is None:
            continue
        arr = tensor.numpy()
        if key == "conv_proj.weight":  # (O, I, H, W) -> (O, H, W, I)
            arr = arr.transpose(0, 2, 3, 1)
        weights.append((mlx_key, mx.array(arr)))

    model.load_weights(weights)
    mx.eval(model.parameters())
    return model
