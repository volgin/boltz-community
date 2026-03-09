#!/usr/bin/env python3
"""
Patch cuequivariance_ops cache_manager.py to use PyTorch for GPU detection.

The NVML APIs used by cuequivariance_ops for GPU detection are not supported
on consumer GeForce GPUs (pynvml raises NVMLError_NotSupported). When this
happens the code silently falls back to hardcoded NVIDIA RTX A6000 specs,
causing Triton kernels to be tuned for the wrong GPU profile.

This patch makes the fallback use torch.cuda.get_device_properties() to get
accurate GPU specs (name, compute capability, SM count) for any GPU.
"""
import re
import sys


def find_cache_manager() -> "Path | None":
    try:
        import cuequivariance_ops.triton.cache_manager as _cm
    except ImportError:
        return None
    import inspect
    from pathlib import Path
    return Path(inspect.getfile(_cm))


def already_patched(text: str) -> bool:
    """Check if the torch fallback is already present."""
    return "torch.cuda.get_device_properties" in text


def apply_patch(path) -> bool:
    original = path.read_text()

    if already_patched(original):
        return None  # sentinel: already patched

    patched = original

    # ------------------------------------------------------------------
    # Patch 1: get_gpu_name() — use torch as fallback instead of
    # hardcoding "NVIDIA RTX A6000".
    # ------------------------------------------------------------------
    patched = re.sub(
        r'(except Exception as e:\s*\n\s*)print\(f"Error getting GPU memory: \{e\}"\)\s*\n(\s*)return "NVIDIA RTX A6000"  # default',
        r"""\1try:
\2    import torch
\2    if torch.cuda.is_available():
\2        return torch.cuda.get_device_name(0)
\2except Exception:
\2    pass
\2return "NVIDIA RTX A6000"  # default""",
        patched,
    )

    # ------------------------------------------------------------------
    # Patch 2: get_gpu_information() fallback — when the GPU name is not
    # in default_map, construct GPU info from torch device properties
    # instead of blindly using A6000 specs.
    # ------------------------------------------------------------------
    patched = re.sub(
        r'(\s*if gpu_name in default_map:\s*\n\s*defaults = default_map\[gpu_name\]\s*\n\s*)else:\s*\n(\s*)defaults = default_map\["NVIDIA RTX A6000"\]',
        r"""\1else:
\2try:
\2    import torch
\2    if torch.cuda.is_available():
\2        p = torch.cuda.get_device_properties(0)
\2        defaults = {
\2            "name": p.name,
\2            "major": p.major,
\2            "minor": p.minor,
\2            "total_memory": p.total_memory // (1024 ** 3),
\2            "multi_processor_count": p.multi_processor_count,
\2            "power_limit": 0,
\2            "clock_rate": 0,
\2        }
\2    else:
\2        defaults = default_map["NVIDIA RTX A6000"]
\2except Exception:
\2    defaults = default_map["NVIDIA RTX A6000"]""",
        patched,
    )

    if patched == original:
        return False

    path.write_text(patched)
    return True


def main():
    from pathlib import Path

    path = find_cache_manager()
    if path is None:
        print("cuequivariance_ops not installed -- skipping patch")
        sys.exit(0)

    print(f"  Patching: {path}")

    result = apply_patch(path)
    if result is None:
        print("  Already patched (torch.cuda fallback present)")
    elif result:
        print("  GPU detection patch applied successfully")
        print("  Triton kernels will now use correct GPU profile (PyTorch fallback)")
    else:
        print("  WARNING: Patch patterns not matched -- cuequivariance_ops may have changed")
        print("  GPU detection may still use incorrect A6000 fallback profile")
        print(f"  Check manually: {path}")


if __name__ == "__main__":
    main()
