# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained int4 ConvRot W4A4 forward path for AMD RDNA4 (gfx12).

Vendored from comfy-kitchen's HIP backend so this ComfyUI node has no external
dependency. Wires the native HIP int4 WMMA GEMM (via ._loader) to the pure-torch
ConvRot rotation/quant helpers (./eager), bypassing comfy_kitchen's registry.

Public API (what the node calls):
    quantize_convrot_w4a4_weight(weight, convrot_groupsize) -> (qweight_i8, scale_f32)
    convrot_w4a4_linear(x, qweight, wscales, bias, convrot_groupsize) -> tensor
    is_available() -> bool          # gfx12 ROCm + built/loadable kernel
    pick_groupsize(in_features) -> int | None
"""
from __future__ import annotations

import torch

from .eager import (
    _build_hadamard,
    _rotate_activation,
    dequantize_convrot_w4a4_weight,
    quantize_convrot_w4a4_weight,
    quantize_signed_int4_rowwise,
)
from .eager import convrot_w4a4_linear as _eager_convrot_w4a4_linear
from .int8_eager import quantize_signed_int8_rowwise
from .int8_eager import convrot_w8a8_linear as _eager_convrot_w8a8_linear
from ._codec import _INT4_GROUP_SIZE
from .hip_loader import (
    convrot_quant_int4, convrot_quant_int8, int4_convrot_gemm, int8_convrot_gemm,
)
from .hip_loader import is_available as _lib_available

__all__ = [
    "quantize_convrot_w4a4_weight",
    "dequantize_convrot_w4a4_weight",
    "convrot_w4a4_linear",
    "convrot_w8a8_linear",
    "is_available",
    "pick_groupsize",
]

# dtypes the HIP kernel emits natively; others compute in fp32 then cast.
_NATIVE_OUT = frozenset({torch.float32, torch.bfloat16})

# Row count above which the fused HIP rotation+quant beats the torch matmul path.
_FUSED_ROT_MIN_ROWS = 512

# The fused conv-rot kernel stages the whole transformed row in LDS: dynamic
# rowbuf (K bf16 = 2*K bytes) plus two static float[256] scratch arrays
# (g[] + red[] = 2 KiB). Guard the fused path so an unusually wide layer falls
# back to the torch rotate+quant path instead of failing the kernel launch with
# an out-of-LDS hipError.
_FUSED_STATIC_LDS = 2048
_LDS_BUDGET = None


def _fused_rot_lds_fits(k: int) -> bool:
    global _LDS_BUDGET
    if _LDS_BUDGET is None:
        try:
            _LDS_BUDGET = int(torch.cuda.get_device_properties(0).shared_memory_per_block)
        except Exception:
            _LDS_BUDGET = 65536  # RDNA4 (gfx12) LDS per workgroup
    return k * 2 + _FUSED_STATIC_LDS <= _LDS_BUDGET


def pick_groupsize(in_features: int) -> int | None:
    """Largest supported ConvRot groupsize dividing in_features (256/64/16)."""
    for g in (256, 64, 16):
        if in_features % g == 0:
            return g
    return None


def _is_gfx12() -> bool:
    if getattr(torch.version, "hip", None) is None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    except Exception:
        return False
    return arch.startswith("gfx12")


def is_available() -> bool:
    """True only on an RDNA4 (gfx12) ROCm device with a loadable int4 kernel."""
    return _is_gfx12() and _lib_available()


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    """Compute ``x @ W.T + bias`` via the RDNA4 native int4 ConvRot GEMM."""
    if linear_dtype == "int8":
        # int8-forced layers keep the eager reference path.
        return _eager_convrot_w4a4_linear(
            x, qweight, wscales, bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )
    if linear_dtype != "int4":
        raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")
    if x.shape[-1] % convrot_groupsize != 0:
        raise ValueError(f"Input K={x.shape[-1]} not divisible by convrot_groupsize {convrot_groupsize}")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    # The fused (one-block-per-row) HIP rotation wins big once there are enough
    # rows to fill the GPU; below that, torch's batched Hadamard matmul is faster.
    if (convrot_groupsize in (16, 64, 256)
            and x2d.shape[0] >= _FUSED_ROT_MIN_ROWS
            and _fused_rot_lds_fits(x2d.shape[1])):
        qact, x_scale = convrot_quant_int4(x2d, convrot_groupsize)
    else:
        h = _build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
        x_rot = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
        qact, x_scale = quantize_signed_int4_rowwise(x_rot)

    out_dtype = x.dtype if x.dtype in _NATIVE_OUT else torch.float32
    out = int4_convrot_gemm(qact, qweight, x_scale, wscales, bias, out_dtype)
    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    return out.reshape(*orig_shape[:-1], qweight.shape[0])


def convrot_w8a8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """Compute ``x @ W.T + bias`` via the RDNA4 native int8 ConvRot GEMM.

    Rotates + int8-quantizes the activation, then runs the native
    wmma_i32_16x16x16_iu8 kernel (which takes the int8 weight directly, avoiding
    any fp32 weight materialization). Falls back to the eager reference path on
    any failure. qweight [N,K] int8, wscales [N] f32.
    """
    try:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        # Fused native rotate+int8-quant once there are enough rows to fill the
        # GPU; otherwise torch's batched Hadamard + rowwise quant is faster.
        if (convrot_groupsize in (16, 64, 256)
                and x2d.shape[0] >= _FUSED_ROT_MIN_ROWS
                and _fused_rot_lds_fits(x2d.shape[1])):
            qact, x_scale = convrot_quant_int8(x2d, convrot_groupsize)
        else:
            h = _build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
            x_rot = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
            qact, x_scale = quantize_signed_int8_rowwise(x_rot)
        out_dtype = x.dtype if x.dtype in _NATIVE_OUT else torch.float32
        out = int8_convrot_gemm(qact, qweight, x_scale, wscales, bias, out_dtype)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out.reshape(*orig_shape[:-1], qweight.shape[0])
    except Exception:
        return _eager_convrot_w8a8_linear(
            x, qweight, wscales, bias=bias, convrot_groupsize=convrot_groupsize)
