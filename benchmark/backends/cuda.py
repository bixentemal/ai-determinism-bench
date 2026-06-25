from __future__ import annotations

from typing import Any, Callable

import numpy as np

from benchmark.backends.base import Backend


class CUDABackend(Backend):
    NAME = "cuda"

    @staticmethod
    def available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def unavailable_reason() -> str:
        return "torch.cuda.is_available() is False on this host"

    @property
    def max_det_level(self) -> str:
        return "FULL_DET"

    @property
    def device(self) -> str:
        return "cuda"

    def hardware_metadata(self) -> dict[str, Any]:
        import torch

        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "device": torch.cuda.get_device_name(idx),
            "device_index": idx,
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "compute_capability": f"{props.major}.{props.minor}",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }

    def time_call(self, fn: Callable[[], Any]) -> tuple[float, Any]:
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), output

    def to_numpy(self, output: Any) -> np.ndarray:
        import torch

        if isinstance(output, torch.Tensor):
            return output.detach().to(torch.float32).cpu().numpy()
        return np.asarray(output, dtype=np.float32)

    def reset_peak_memory(self) -> None:
        import torch

        torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float | None:
        import torch

        return torch.cuda.max_memory_allocated() / (1024 * 1024)
