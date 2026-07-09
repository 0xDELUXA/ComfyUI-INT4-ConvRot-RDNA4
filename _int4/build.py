# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the int4 convrot GEMM shared library with hipcc.

Standalone (no CMake / torch cpp_extension) so it works on Windows+ROCm where
those paths are flaky. Called lazily by the loader if the built lib is missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "int4_gemm.hip")


def lib_name() -> str:
    return "ck_int4_gemm.dll" if os.name == "nt" else "libck_int4_gemm.so"


def lib_path() -> str:
    return os.path.join(_HERE, lib_name())


def src_path() -> str:
    return _SRC


def is_stale() -> bool:
    """True if the built lib is missing or older than the .hip source (so an
    edited kernel is picked up on next load instead of running a stale binary)."""
    out = lib_path()
    if not os.path.exists(out):
        return True
    try:
        return os.path.getmtime(_SRC) > os.path.getmtime(out)
    except OSError:
        return True


def _find_hipcc() -> str | None:
    hc = shutil.which("hipcc")
    if hc:
        return hc
    cand = os.path.join(
        sys.prefix, "Lib", "site-packages", "_rocm_sdk_devel", "bin",
        "hipcc.exe" if os.name == "nt" else "hipcc",
    )
    return cand if os.path.exists(cand) else None


def gpu_arch() -> str:
    """The active ROCm device architecture (its gcnArchName, minus any :xnack
    suffix). Raises if it cannot be detected, rather than guessing a specific
    arch: building with --offload-arch for the wrong GPU yields a kernel that
    will not run."""
    try:
        import torch
        name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    except Exception:
        name = None
    if not name:
        raise RuntimeError(
            "could not detect the ROCm GPU architecture from "
            "torch.cuda.get_device_properties(0).gcnArchName; pass the architecture "
            "string explicitly to build(arch=...)"
        )
    return name.split(":")[0]


def build(arch: str | None = None, verbose: bool = True) -> str:
    hipcc = _find_hipcc()
    if not hipcc:
        raise RuntimeError("hipcc not found on PATH; cannot build the HIP int4 GEMM lib")
    arch = arch or gpu_arch()
    out = lib_path()
    cmd = [hipcc, "-O3", f"--offload-arch={arch}", "-shared", _SRC, "-o", out]
    if verbose:
        print("building:", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True)
    # device pass emits a harmless dllexport-ignored warning; only fail on error.
    if cp.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"hipcc build failed:\n{cp.stderr}")
    return out


if __name__ == "__main__":
    print("built:", build())
