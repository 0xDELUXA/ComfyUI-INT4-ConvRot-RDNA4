# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signed-int4 nibble pack/unpack codec and quantizer constants.

Vendored verbatim from comfy-kitchen (comfy_kitchen/backends/eager/svdquant.py)
so this node is self-contained. See NOTICE for attribution.
"""
from __future__ import annotations

import torch

# ConvRot int4 quantizer contract (must match the HIP kernel):
#   symmetric absmax quantization, scale = max / _INT4_MAX, clamp [-7, 7].
# -8 is representable in the nibble but never emitted, to keep dequant symmetric.
_INT4_GROUP_SIZE = 64
_INT4_MAX = 7   # signed quantizer range: [-7, 7], scale = max/7


def _pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Pack (..., K) int4 values into (..., K // 2) int8 (low = even column)."""
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_row_major with signed-nibble interpretation ([-8, 7])."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)
