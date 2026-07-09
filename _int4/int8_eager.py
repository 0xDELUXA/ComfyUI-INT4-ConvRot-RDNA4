"""Eager (pure-torch) ConvRot W8A8 int8 path - the mixed-precision middle tier.

Sits between int4 (smallest, most error) and bf16 (largest, no quant error): an
int8 weight is half the size of bf16 but far more accurate than int4, so spending
a VRAM budget on int8 protects ~2x more layers than spending it on bf16.

Reuses the same regular-Hadamard ConvRot rotation as the int4 path (from .eager)
so the scheme is identical apart from the quant width (8-bit, scale = max/127) and
the storage (plain int8 [N, K], no nibble packing). This forward is the reference
and fallback (fp32 accumulation); the fast path is the native int8 WMMA kernel in
convrot_w8a8_linear, used automatically when the HIP backend is available.
"""
from __future__ import annotations

import torch

from .eager import _build_hadamard, _rotate_activation, _rotate_weight

_INT8_MAX = 127  # signed int8 quant range [-127, 127], scale = absmax / 127


def quantize_signed_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row absmax int8 quantization. Returns (int8 [R, C], scale [R] f32)."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT8_MAX
    q = (x / scales).round_().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scales.reshape(x.shape[0]).to(torch.float32)


def quantize_convrot_w8a8_weight(
    weight: torch.Tensor,
    convrot_groupsize: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a weight with ConvRot and quantize it to signed int8 (per-row).

    Returns (qweight int8 [N, K], scale f32 [N]). Unlike int4, the weight is NOT
    nibble-packed, so the stored tensor's column count equals in_features.
    """
    if weight.dim() != 2:
        raise ValueError(f"ConvRot W8A8 expects a 2D tensor, got {tuple(weight.shape)}")
    if weight.shape[-1] % convrot_groupsize != 0:
        raise ValueError(f"in_features {weight.shape[-1]} not divisible by "
                         f"convrot_groupsize {convrot_groupsize}")
    h = _build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    w_rot = _rotate_weight(weight, h, convrot_groupsize)
    return quantize_signed_int8_rowwise(w_rot)


def dequantize_convrot_w8a8_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize int8 ConvRot weights and rotate back to the original basis."""
    w_rot = qdata.to(torch.float32) * scales.to(qdata.device, torch.float32).reshape(-1, 1)
    h = _build_hadamard(convrot_groupsize, device=qdata.device, dtype=torch.float32)
    return _rotate_weight(w_rot, h, convrot_groupsize).to(output_dtype)


def convrot_w8a8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """Compute ``x @ W.T + bias`` with online int8 activation quant (W8A8 ConvRot).

    Rotates and int8-quantizes the activation the same way the weight was, then
    does an int8 matmul accumulated in fp32 and rescales by the row/col scales.
    """
    if x.shape[-1] != qweight.shape[-1]:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1]}")
    if x.shape[-1] % convrot_groupsize != 0:
        raise ValueError(f"Input K={x.shape[-1]} not divisible by convrot_groupsize {convrot_groupsize}")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    h = _build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
    x_rot = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
    qact, x_scale = quantize_signed_int8_rowwise(x_rot)

    # fp32 accumulation of the int8 products (reference path; no native int GEMM).
    out = qact.to(torch.float32) @ qweight.to(torch.float32).t()
    out = out * x_scale.reshape(-1, 1) * wscales.to(out.device, torch.float32).reshape(1, -1)
    if bias is not None:
        out = out + bias.to(out.device, torch.float32).reshape(1, -1)
    return out.to(x.dtype).reshape(*orig_shape[:-1], qweight.shape[0])
