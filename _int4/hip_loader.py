# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ctypes loader for the gfx12 int4 convrot GEMM shared library."""
from __future__ import annotations

import ctypes
import os

import torch

from .build import build, is_stale, lib_path

_lib = None
_fn = None
_fn_splitk = None
_fn_quant = None
_fn_quant8 = None
_fn_int8 = None

# 128x128 output tiles. Split K only when the output-tile grid is genuinely too
# small to fill the GPU (a handful of tiles) AND K is large. Outside that narrow
# regime the single-pass kernel already saturates and the split path's fp32
# workspace + finalize just add overhead, so the setting stays conservative.
_TILE = 128
_SPLITK_MAX_TILES = 12       # only split when ntiles is at or below this
_SPLITK_TARGET_BLOCKS = 64   # grow the grid toward ~2 waves on the target GPU
_SPLITK_MAX = 8


def _choose_nsplit(M, N, K):
    ntiles = ((M + _TILE - 1) // _TILE) * ((N + _TILE - 1) // _TILE)
    nchunk = K // 32
    if ntiles > _SPLITK_MAX_TILES or nchunk < 24:
        return 1
    nsplit = min(_SPLITK_TARGET_BLOCKS // max(ntiles, 1), nchunk // 8, _SPLITK_MAX)
    return max(nsplit, 1)

_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1}
_IN_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def _ensure_loaded():
    global _lib, _fn, _fn_quant, _fn_quant8, _fn_int8
    if _fn is not None:
        return
    path = lib_path()
    if is_stale():
        build()
    _lib = ctypes.CDLL(path)
    _fn = _lib.ck_int4_convrot_gemm
    _fn.restype = ctypes.c_int
    _fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    global _fn_splitk
    _fn_splitk = _lib.ck_int4_convrot_gemm_splitk
    _fn_splitk.restype = ctypes.c_int
    _fn_splitk.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _fn_quant = _lib.ck_convrot_quant_int4
    _fn_quant.restype = ctypes.c_int
    _fn_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _fn_int8 = _lib.ck_int8_convrot_gemm
    _fn_int8.restype = ctypes.c_int
    _fn_int8.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _fn_quant8 = _lib.ck_convrot_quant_int8
    _fn_quant8.restype = ctypes.c_int
    _fn_quant8.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]


def convrot_quant_int4(x, convrot_groupsize):
    """Fused ConvRot rotation + rowwise int4 quantize. Returns (qpacked[M,K/2] int8,
    scale[M] f32). Only groupsize 256 is supported; raises otherwise."""
    _ensure_loaded()
    if convrot_groupsize not in (16, 64, 256):
        raise ValueError("fused convrot_quant_int4 supports groupsize 16/64/256")
    x = x.contiguous()
    M, K = x.shape
    qout = torch.empty((M, K // 2), dtype=torch.int8, device=x.device)
    scale = torch.empty((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    rc = _fn_quant(
        ctypes.c_void_p(x.data_ptr()), _IN_DTYPE_CODE[x.dtype],
        ctypes.c_void_p(qout.data_ptr()), ctypes.c_void_p(scale.data_ptr()),
        M, K, convrot_groupsize, ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"ck_convrot_quant_int4 failed, hipError={rc}")
    return qout, scale


def convrot_quant_int8(x, convrot_groupsize):
    """Fused ConvRot rotation + rowwise int8 quantize. Returns (q[M,K] int8,
    scale[M] f32). Supports groupsize 16/64/256; raises otherwise."""
    _ensure_loaded()
    if convrot_groupsize not in (16, 64, 256):
        raise ValueError("fused convrot_quant_int8 supports groupsize 16/64/256")
    x = x.contiguous()
    M, K = x.shape
    qout = torch.empty((M, K), dtype=torch.int8, device=x.device)
    scale = torch.empty((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    rc = _fn_quant8(
        ctypes.c_void_p(x.data_ptr()), _IN_DTYPE_CODE[x.dtype],
        ctypes.c_void_p(qout.data_ptr()), ctypes.c_void_p(scale.data_ptr()),
        M, K, convrot_groupsize, ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"ck_convrot_quant_int8 failed, hipError={rc}")
    return qout, scale


def int8_convrot_gemm(x_int8, w_int8, x_scale, w_scale, bias, out_dtype):
    """out[M,N] = (X[M,K] @ W[N,K]^T) * x_scale[:,None] * w_scale[None,:] (+bias).

    x_int8 [M,K] int8, w_int8 [N,K] int8 (plain, unpacked); x_scale [M] f32,
    w_scale [N] f32, bias [N] or None. Native wmma_i32_16x16x16_iu8 path.
    """
    _ensure_loaded()
    if out_dtype not in _DTYPE_CODE:
        raise ValueError(f"hip int8 gemm output dtype must be float32/bfloat16, got {out_dtype}")
    x_int8 = x_int8.contiguous()
    w_int8 = w_int8.contiguous()
    M, K = x_int8.shape
    N, Kw = w_int8.shape
    if K != Kw:
        raise ValueError(f"K mismatch: x {x_int8.shape} w {w_int8.shape}")
    x_scale = x_scale.to(torch.float32).contiguous()
    w_scale = w_scale.to(torch.float32).contiguous()
    out = torch.empty((M, N), dtype=out_dtype, device=x_int8.device)
    bias_ptr = 0
    if bias is not None:
        bias = bias.to(torch.float32).contiguous()
        bias_ptr = bias.data_ptr()
    stream = torch.cuda.current_stream().cuda_stream
    rc = _fn_int8(
        ctypes.c_void_p(x_int8.data_ptr()), ctypes.c_void_p(w_int8.data_ptr()),
        ctypes.c_void_p(x_scale.data_ptr()), ctypes.c_void_p(w_scale.data_ptr()),
        ctypes.c_void_p(bias_ptr), ctypes.c_void_p(out.data_ptr()),
        M, N, K, _DTYPE_CODE[out_dtype], ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"ck_int8_convrot_gemm failed, hipError={rc}")
    return out


def is_available() -> bool:
    try:
        _ensure_loaded()
        return True
    except Exception:
        return False


def int4_convrot_gemm(x_packed, w_packed, x_scale, w_scale, bias, out_dtype):
    """out[M,N] = (Xint[M,K] @ Wint[N,K]^T) * x_scale[:,None] * w_scale[None,:] (+bias).

    x_packed [M,K/2] int8, w_packed [N,K/2] int8 (comfy _pack_int4_row_major);
    x_scale [M] f32, w_scale [N] f32, bias [N] or None.
    """
    _ensure_loaded()
    if out_dtype not in _DTYPE_CODE:
        raise ValueError(f"hip int4 gemm output dtype must be float32/bfloat16, got {out_dtype}")
    x_packed = x_packed.contiguous()
    w_packed = w_packed.contiguous()
    M, Kp = x_packed.shape
    N, Kp2 = w_packed.shape
    if Kp != Kp2:
        raise ValueError(f"K mismatch: x {x_packed.shape} w {w_packed.shape}")
    x_scale = x_scale.to(torch.float32).contiguous()
    w_scale = w_scale.to(torch.float32).contiguous()
    K = Kp * 2
    stream = torch.cuda.current_stream().cuda_stream

    nsplit = _choose_nsplit(M, N, K)
    if nsplit > 1:
        # Split-K: accumulate scaled partials into a zeroed fp32 buffer, then
        # add bias + cast in torch.
        accum = torch.zeros((M, N), dtype=torch.float32, device=x_packed.device)
        rc = _fn_splitk(
            ctypes.c_void_p(x_packed.data_ptr()), ctypes.c_void_p(w_packed.data_ptr()),
            ctypes.c_void_p(x_scale.data_ptr()), ctypes.c_void_p(w_scale.data_ptr()),
            ctypes.c_void_p(accum.data_ptr()), M, N, K, nsplit, ctypes.c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"ck_int4_convrot_gemm_splitk failed, hipError={rc}")
        if bias is not None:
            accum = accum + bias.to(torch.float32)
        return accum.to(out_dtype)

    out = torch.empty((M, N), dtype=out_dtype, device=x_packed.device)
    bias_ptr = 0
    if bias is not None:
        bias = bias.to(torch.float32).contiguous()
        bias_ptr = bias.data_ptr()
    rc = _fn(
        ctypes.c_void_p(x_packed.data_ptr()), ctypes.c_void_p(w_packed.data_ptr()),
        ctypes.c_void_p(x_scale.data_ptr()), ctypes.c_void_p(w_scale.data_ptr()),
        ctypes.c_void_p(bias_ptr), ctypes.c_void_p(out.data_ptr()),
        M, N, K, _DTYPE_CODE[out_dtype], ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"ck_int4_convrot_gemm failed, hipError={rc}")
    return out
