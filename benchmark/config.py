from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """Everything a run needs, resolved from CLI flags (SPEC §CLI Design)."""

    tier: str = "core"
    classes: list[str] | None = None
    presets: list[str] = field(default_factory=lambda: ["production", "strict"])
    dtype: dict[str, str] = field(default_factory=lambda: {"vision": "fp32", "llm": "bf16"})
    n_runs: int = 50
    n_warmup: int = 10
    seed: int = 1234

    # LLM
    prompt_len: int = 512
    decode_tokens: int = 128
    decode_repeats: int = 5  # runs used for token_repro_rate / decode timing

    # Vision
    batch_size: int = 32
    image_size: int = 224

    backend_name: str | None = None
    output: str | None = None
    formats: list[str] = field(default_factory=lambda: ["table"])

    @classmethod
    def quick(cls) -> "RunConfig":
        """Fast smoke profile: small shapes + few runs (SPEC: --quick)."""
        return cls(
            tier="quick",
            n_runs=5,
            n_warmup=2,
            prompt_len=64,
            decode_tokens=16,
            decode_repeats=3,
            batch_size=8,  # smaller vision batch keeps the smoke test snappy
        )

    def dtype_for(self, model_class: str) -> str:
        return self.dtype.get(model_class, "fp32")
