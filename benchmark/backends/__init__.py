"""Hardware backends. CPU + MLX are runnable on Apple Silicon; CUDA is a stub
until it can be implemented and validated on a GPU host (see SPEC §Hardware Backends)."""

from benchmark.backends.base import Backend
from benchmark.backends.cpu import CPUBackend
from benchmark.backends.mlx import MLXBackend
from benchmark.backends.cuda import CUDABackend

# Auto-detection order: prefer accelerated backends, fall back to CPU.
_REGISTRY = [CUDABackend, MLXBackend, CPUBackend]


def available_backends() -> list[str]:
    return [b.NAME for b in _REGISTRY if b.available()]


def get_backend(name: str | None) -> Backend:
    """Return a backend by name, or auto-detect the best available one."""
    by_name = {b.NAME: b for b in _REGISTRY}
    if name is not None:
        if name not in by_name:
            raise ValueError(f"Unknown backend {name!r}; choose from {list(by_name)}")
        backend = by_name[name]
        if not backend.available():
            raise SystemExit(
                f"Backend {name!r} is not available on this host "
                f"({backend.unavailable_reason()})."
            )
        return backend()
    for backend in _REGISTRY:
        if backend.available():
            return backend()
    raise SystemExit("No usable backend found.")
