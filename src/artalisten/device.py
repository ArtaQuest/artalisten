"""Pick a compute device and release it between pipeline stages."""

from __future__ import annotations

import gc
import os


def unified_memory_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page) / (1024**3)
    except (OSError, ValueError, AttributeError):
        return 0.0


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pick_device(requested: str) -> str:
    """Return ``cuda``, ``mps``, or ``cpu``. ``auto`` prefers CUDA, then MPS, then CPU."""
    choice = (requested or "auto").lower()
    if choice not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"unknown device {requested!r}; use auto, cpu, mps, or cuda")
    if choice == "auto":
        if cuda_available():
            return "cuda"
        return "mps" if mps_available() else "cpu"
    if choice == "mps" and not mps_available():
        raise RuntimeError("MPS was requested but this Mac cannot use it")
    if choice == "cuda" and not cuda_available():
        raise RuntimeError("CUDA was requested but no GPU is available")
    return choice


def torch_device(name: str):
    import torch

    return torch.device(name)


def release_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
