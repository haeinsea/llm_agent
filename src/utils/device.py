from __future__ import annotations

from typing import Any


def get_torch_device(prefer_mps: bool = True) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if prefer_mps and mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def torch_device_info(prefer_mps: bool = True) -> dict[str, Any]:
    info = {
        "selected_device": "cpu",
        "cuda_available": False,
        "mps_built": False,
        "mps_available": False,
    }
    try:
        import torch
    except Exception:
        return info

    mps_backend = getattr(torch.backends, "mps", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["mps_built"] = bool(mps_backend is not None and mps_backend.is_built())
    info["mps_available"] = bool(mps_backend is not None and mps_backend.is_available())
    info["selected_device"] = get_torch_device(prefer_mps=prefer_mps)
    return info


def synchronize_torch_device(device: str) -> None:
    try:
        import torch
    except Exception:
        return

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps":
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "synchronize"):
            mps_backend.synchronize()
