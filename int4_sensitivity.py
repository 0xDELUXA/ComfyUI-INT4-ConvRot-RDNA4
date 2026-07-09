"""Calibration-free per-layer quantization sensitivity scorer.

Measures, for every Linear weight that WOULD be quantized, how much the int4
ConvRot round-trip moves the weight:

    rel_err = || W - dequant(quant(W)) ||_F  /  || W ||_F

This is data-free (no calibration inputs, no forward passes): it reuses the exact
quantize/dequantize path the real loader uses, so the error it reports is the
error the model will actually see. Ranking layers by this score is the raw signal
a future mixed-precision selector uses to decide which layers to keep in higher
precision (int8/bf16) instead of int4.

This scorer is measurement only: it logs a ranking and changes NO selection
behavior. It is opt-in (the loader calls it when asked) so it never slows a
normal load.
"""
from __future__ import annotations

import torch

# Byte cost per parameter for each precision (weight payload only; the tiny
# per-row f32 scale is added separately). Used for the impact-per-byte column
# that a later knapsack selector will rank on.
_BYTES_PER_PARAM = {"int4": 0.5, "int8": 1.0, "bf16": 2.0}


def _pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def score_weight(w: torch.Tensor, gs: int):
    """Frobenius error of the int4 ConvRot round-trip for one weight.

    Returns (rel_err, abs_err): the relative error ``||W-deq||/||W||`` (for the
    human-readable ranking) and the absolute error ``||W-deq||`` (which the
    knapsack selector uses as the "how much quality is lost" quantity, since it
    scales with the layer's magnitude/size, i.e. its real effect on the output).
    Returns None if the weight cannot go through the int4 path (e.g. in_features
    not divisible by the fixed quant group size), i.e. it would fall back to bf16
    at load time anyway.
    """
    from ._int4.eager import (  # lazy: backend may be unavailable at import time
        quantize_convrot_w4a4_weight,
        dequantize_convrot_w4a4_weight,
    )

    dev = _pick_device()
    with torch.no_grad():
        w32 = w.detach().to(dev, torch.float32)
        try:
            qw, ws = quantize_convrot_w4a4_weight(
                w32.to(torch.float16), convrot_groupsize=gs)
            deq = dequantize_convrot_w4a4_weight(
                qw, ws, convrot_groupsize=gs, output_dtype=torch.float32)
        except Exception:
            return None
        abs_err = torch.linalg.norm(w32 - deq).item()
        denom = torch.linalg.norm(w32).item()
        rel_err = 0.0 if denom == 0.0 else abs_err / denom
        return rel_err, abs_err


def score_weight_int8(w: torch.Tensor, gs: int):
    """Absolute Frobenius error of the int8 ConvRot round-trip for one weight
    (the int8 tier's quality loss). Returns None if the shape is unsupported."""
    from ._int4.int8_eager import (  # lazy: backend may be unavailable at import
        quantize_convrot_w8a8_weight,
        dequantize_convrot_w8a8_weight,
    )
    dev = _pick_device()
    with torch.no_grad():
        w32 = w.detach().to(dev, torch.float32)
        try:
            qw, ws = quantize_convrot_w8a8_weight(w32.to(torch.float16), convrot_groupsize=gs)
            deq = dequantize_convrot_w8a8_weight(qw, ws, convrot_groupsize=gs,
                                                 output_dtype=torch.float32)
        except Exception:
            return None
        return torch.linalg.norm(w32 - deq).item()


def score_layer_output(w: torch.Tensor, gs: int, probe_rows: int = 256):
    """Activation-aware drift: how much the layer's *output* moves under the full
    W4A4 / W8A8 pipeline (weight AND activation quantization), on probe activations.

    Runs the real quantized linear on random rotated-Gaussian inputs and compares
    to the true bf16 output. This captures activation-quant sensitivity that the
    weight-only round-trip misses. Because ConvRot rotates activations to be
    near-whitened before quantizing, a Gaussian probe is representative of the
    real rotated-activation distribution, so no calibration data is needed.

    Returns (rel_err_int4, abs_err_int4, abs_err_int8) or None on unsupported shape.
    """
    from ._int4 import (  # lazy: backend may be unavailable at import time
        convrot_w4a4_linear, convrot_w8a8_linear, quantize_convrot_w4a4_weight,
    )
    from ._int4.int8_eager import quantize_convrot_w8a8_weight

    dev = _pick_device()
    with torch.no_grad():
        w16 = w.detach().to(dev, torch.float16)
        K = w16.shape[1]
        try:
            x = torch.randn(probe_rows, K, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ w16.float().t()
            qw4, ws4 = quantize_convrot_w4a4_weight(w16, convrot_groupsize=gs)
            o4 = convrot_w4a4_linear(x, qw4, ws4, convrot_groupsize=gs).float()
            qw8, ws8 = quantize_convrot_w8a8_weight(w16, gs)
            o8 = convrot_w8a8_linear(x, qw8, ws8, convrot_groupsize=gs).float()
        except Exception:
            return None
        denom = torch.linalg.norm(ref).item()
        abs4 = torch.linalg.norm(o4 - ref).item()
        abs8 = torch.linalg.norm(o8 - ref).item()
        rel4 = 0.0 if denom == 0.0 else abs4 / denom
        return rel4, abs4, abs8


def rank_layers(sd, min_in_features, groupsize_fn, excluded_names=None, include_int8=False,
                scoring_mode="weight"):
    """Score every quantizable Linear weight in a state_dict.

    Mirrors Int4ConvRotOps._should_quantize eligibility: 2D weight, in_features
    >= min_in_features, and a valid ConvRot groupsize. Returns a list of records
    sorted worst (most sensitive) first. `excluded_names` (the current static
    skip list) is used only to annotate whether today's heuristic would already
    keep a layer at bf16 - it does NOT affect scoring.
    """
    excluded_names = excluded_names or []
    records = []
    for key, w in sd.items():
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or not key.endswith(".weight"):
            continue
        out_f, in_f = w.shape
        if in_f < min_in_features:
            continue
        gs = groupsize_fn(in_f)
        if gs is None:
            continue
        if scoring_mode == "output":
            scored = score_layer_output(w, gs)
            if scored is None:
                continue
            rel_err, abs_err, abs_err_int8 = scored  # int8 always measured here
        else:
            scored = score_weight(w, gs)
            if scored is None:
                continue  # would fall back to bf16 (int4 path rejects this shape)
            rel_err, abs_err = scored
            abs_err_int8 = score_weight_int8(w, gs) if include_int8 else None
        prefix = key[: -len(".weight")]
        params = out_f * in_f
        records.append({
            "key": prefix,
            "out_features": out_f,
            "in_features": in_f,
            "groupsize": gs,
            "params": params,
            "rel_err": rel_err,
            "abs_err": abs_err,          # int4 round-trip absolute error
            "abs_err_int8": abs_err_int8,  # int8 round-trip absolute error (or None)
            # Extra bytes it costs to keep this layer bf16 instead of int4
            # (the knapsack's price): (2.0 - 0.5) bytes/param.
            "bf16_extra_bytes": params * (_BYTES_PER_PARAM["bf16"] - _BYTES_PER_PARAM["int4"]),
            "excluded": any(n in prefix for n in excluded_names),
        })
    records.sort(key=lambda r: r["rel_err"], reverse=True)
    return records


def format_ranking(records, top=None) -> str:
    """Human-readable ranking table for the log."""
    if not records:
        return "INT4 ConvRot sensitivity: no quantizable Linear layers found."

    errs = [r["rel_err"] for r in records]
    mean_err = sum(errs) / len(errs)
    n_excluded = sum(1 for r in records if r["excluded"])

    lines = [
        f"INT4 ConvRot sensitivity (calibration-free int4 round-trip): "
        f"{len(records)} quantizable layer(s), mean rel_err={mean_err:.4f}, "
        f"{n_excluded} currently kept bf16 by the name heuristic.",
        "  rank  rel_err   params    K x N            keep-bf16?  layer",
    ]
    shown = records if top is None else records[:top]
    for i, r in enumerate(shown):
        lines.append(
            f"  {i + 1:>4}  {r['rel_err']:.5f}  {r['params'] / 1e6:>6.1f}M  "
            f"{r['in_features']:>5} x {r['out_features']:<6}  "
            f"{'  (bf16)' if r['excluded'] else '        '}    {r['key']}"
        )
    if top is not None and len(records) > top:
        lines.append(f"  ... {len(records) - top} more (worst-first; pass top=None to see all)")

    # Surface the tension: layers the heuristic protects that score LOW (maybe
    # wasted budget) and layers it quantizes that score HIGH (maybe hurting).
    protected_but_low = [r for r in records if r["excluded"]]
    if protected_but_low:
        worst_protected_rank = min(records.index(r) for r in protected_but_low) + 1
        lines.append(
            f"  note: most-sensitive bf16-kept layer ranks #{worst_protected_rank}; "
            f"any quantized layer ranking above it is a candidate to protect instead."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Automatic int4 / int8 / bf16 selection (knapsack).
# ---------------------------------------------------------------------------

_KEY_PREFIXES = ("diffusion_model.", "model.diffusion_model.", "model.", "transformer.")

# Extra bytes/param each upgrade costs, relative to the int4 baseline.
_UPGRADE_COST = {"int8": 1.0 - 0.5, "bf16": 2.0 - 1.0}  # int4->int8, int8->bf16


def normalize_key(key: str) -> str:
    """Strip container prefixes so a state_dict key matches the module namespace
    that Int4ConvRotOps.excluded_exact / int8_exact are tested against."""
    for p in _KEY_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def select_precision(records, extra_vram_bytes):
    """Greedy 3-tier knapsack: spend a VRAM budget (extra bytes over the pure-int4
    baseline) promoting the most-damaged layers up the precision ladder for the
    most quality per byte.

    Every eligible layer starts at int4. Two upgrades are available per layer:
      int4 -> int8: costs 0.5 bytes/param, recovers (int4_err - int8_err) of error
      int8 -> bf16: costs 1.0 bytes/param, recovers the remaining int8_err
    Each step applies whichever affordable upgrade (across all layers) buys the
    most error-reduction per byte, until nothing more fits the budget. int8 is
    typically far cheaper-per-quality than bf16, so the budget stretches further.

    Layers already protected by the name heuristic (excluded) are skipped. Returns
    (bf16_records, int8_records, bytes_spent), each list most-damaged first.
    """
    if extra_vram_bytes <= 0:
        return [], [], 0.0

    # tier 0=int4, 1=int8, 2=bf16. Only score-able (int8 measured) candidates.
    state = [{"r": r, "tier": 0} for r in records
             if not r["excluded"] and r["params"] > 0 and r.get("abs_err_int8") is not None]

    def next_upgrade(s):
        """(cost_bytes, error_reduction) for this layer's next tier step, or None."""
        r = s["r"]
        if s["tier"] == 0:  # int4 -> int8
            return r["params"] * _UPGRADE_COST["int8"], r["abs_err"] - r["abs_err_int8"]
        if s["tier"] == 1:  # int8 -> bf16
            return r["params"] * _UPGRADE_COST["bf16"], r["abs_err_int8"]
        return None

    spent = 0.0
    while True:
        best, best_eff = None, 0.0
        for s in state:
            u = next_upgrade(s)
            if u is None:
                continue
            cost, gain = u
            if cost <= 0 or gain <= 0 or spent + cost > extra_vram_bytes:
                continue
            eff = gain / cost
            if eff > best_eff:
                best, best_eff, best_cost = s, eff, cost
        if best is None:
            break
        best["tier"] += 1
        spent += best_cost

    bf16 = [s["r"] for s in state if s["tier"] == 2]
    int8 = [s["r"] for s in state if s["tier"] == 1]
    bf16.sort(key=lambda r: r["abs_err"], reverse=True)
    int8.sort(key=lambda r: r["abs_err"], reverse=True)
    return bf16, int8, spent
