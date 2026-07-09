"""INT4 ConvRot-W4A4 custom operations for ComfyUI (AMD RDNA4 / gfx12).

Quantizes a diffusion model's Linear layers to int4 ConvRot-W4A4 (on the fly, or
loaded prequantized) and runs them through the vendored native HIP int4 kernel
(see _int4/). Reuses ComfyUI's custom_operations mechanism and per-model
sensitive-layer skip lists for mixed precision.
"""
import logging

import torch
import torch.nn as nn

# Self-contained vendored int4 ConvRot W4A4 forward path (no comfy_kitchen dep).
_CK_OK = False
try:
    from . import _int4 as _ck
    from ._int4 import pick_groupsize as _pick_groupsize
    _CK_OK = True
except Exception as e:  # pragma: no cover
    logging.error(f"INT4 ConvRot: vendored _int4 backend import failed: {e}")

    def _pick_groupsize(in_features):  # fallback so module still imports
        return None

try:
    from comfy.ops import manual_cast
    _COMFY_OPS = True
except Exception:  # pragma: no cover
    _COMFY_OPS = False

# Base class for LoRA weight adapters. Used only to tell apart adapters that
# provide a real additive bypass path h(x) (LoRA/LoHa/LoKr) from ones that don't
# (raw diff/set patches, or OFT/BOFT that only transform the output via g()).
try:
    from comfy.weight_adapter.base import WeightAdapterBase as _WA_BASE
except Exception:  # pragma: no cover
    _WA_BASE = None


def hip_backend_ready() -> bool:
    return _CK_OK and _ck.is_available()


if _COMFY_OPS and _CK_OK:

    class Int4ConvRotOps(manual_cast):
        """ComfyUI ops that quantize Linear weights to int4 ConvRot-W4A4."""

        excluded_names = []      # substrings of layer keys to keep at bf16 (name heuristic)
        excluded_exact = set()   # exact normalized keys to keep at bf16 (auto-selected)
        int8_exact = set()       # exact normalized keys to quantize int8 (auto-selected)
        enable = True            # master toggle
        min_in_features = 512    # skip tiny linears (overhead > int4 win)

        # --- LoRA baking (folded into the bf16 weight before int4 quant) ---
        lora_patches = {}        # {normalized_layer_key: [(v, offset, function, strength), ...]}
        lora_mode = "None"       # "None" (round) | "Stochastic" (stochastic int4 rounding)
        lora_dynamic = False     # True -> keep the LoRA as a runtime side-path on
                                 # quantized layers (not baked); lets strengths change
                                 # without re-quantizing. bf16-kept layers still bake.
        skeleton_meta_init = False  # transient: build meta-device skeletons for key-map discovery
        _applied_lora = set()    # tracks which lora keys actually matched a layer
        _STOCHASTIC_SEED = 1234  # fixed seed -> reproducible stochastic rounding

        class Linear(manual_cast.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                if getattr(Int4ConvRotOps, "skeleton_meta_init", False):
                    # Fast meta-device skeleton (no real weights) used only to
                    # discover the LoRA key map; never sampled from.
                    nn.Module.__init__(self)
                    self.in_features = in_features
                    self.out_features = out_features
                    kw = {"device": "meta"}
                    if dtype is not None:
                        kw["dtype"] = dtype
                    self.weight = nn.Parameter(
                        torch.empty((out_features, in_features), **kw), requires_grad=False)
                    self.bias = (nn.Parameter(torch.empty((out_features,), **kw), requires_grad=False)
                                 if bias else None)
                else:
                    super().__init__(in_features, out_features, bias, device, dtype)
                self._q = None       # quantized weight, int8 dtype (on cuda):
                                     #   int4 -> packed [N, K/2] (two nibbles/byte)
                                     #   int8 -> plain  [N, K]
                self._qs = None      # per-row scale [N] f32
                self._qgs = None     # convrot groupsize used
                self._qmode = None   # "int4" | "int8" (None = not quantized, bf16)
                self._dyn_lora = None  # prepared bypass adapters applied in forward()
                self.comfy_cast_weights = False

            @staticmethod
            def _normalize_lora_key(key):
                if not isinstance(key, str):
                    return key
                for p in ("diffusion_model.", "model.diffusion_model.", "model.", "transformer."):
                    if key.startswith(p):
                        return key[len(p):]
                return key

            @staticmethod
            def _format_lora_patches(patches):
                # -> the (strength, v, key_strength, offset, function) tuples
                # that comfy.lora.calculate_weight expects.
                formatted = []
                for patch in patches or []:
                    if len(patch) == 4:
                        v, offset, function, strength = patch
                    else:
                        v, offset, function = patch
                        strength = 1.0
                    formatted.append((strength, v, 1.0, offset, function))
                return formatted

            def _apply_lora(self, tensor, key):
                """Fold any LoRA patches for `key` into `tensor` (bf16) and return it."""
                if tensor is None or tensor.dtype == torch.int8 or not Int4ConvRotOps.lora_patches:
                    return tensor
                nk = self._normalize_lora_key(key)
                patches = Int4ConvRotOps.lora_patches.get(nk)
                if not patches:
                    return tensor
                import comfy.lora
                import comfy.model_management
                dev = tensor.device
                if dev.type == "cpu" and torch.cuda.is_available():
                    dev = torch.device("cuda")
                tdtype = comfy.model_management.lora_compute_dtype(dev)
                t = tensor.to(device=dev, dtype=tdtype)
                out = comfy.lora.calculate_weight(self._format_lora_patches(patches), t, key)
                Int4ConvRotOps._applied_lora.add(nk)
                return out.to(dtype=tensor.dtype)

            @staticmethod
            def _dyn_capable(v):
                """True if adapter `v` supports the weight-free additive bypass path
                (h(x) = up(down(x)) * scale). Excludes raw diff/set tuple patches,
                DoRA (needs the base weight), and OFT/BOFT (only transform via g())."""
                if not hasattr(v, "weights") or not hasattr(v, "h"):
                    return False
                # Only the default base h() returns zeros -> nothing to add.
                if _WA_BASE is not None and type(v).h is _WA_BASE.h:
                    return False
                w = getattr(v, "weights", None)
                if (getattr(v, "name", None) == "lora"
                        and isinstance(w, (tuple, list)) and len(w) > 4 and w[4] is not None):
                    return False  # dora_scale present
                return True

            def _prepare_dyn_adapters(self, patches):
                """Turn baked-style patches into forward-time bypass adapters for a
                Linear layer. Moves the low-rank tensors to cuda/fp16 once and sets
                the attributes h() reads. Returns None if any patch can't be applied
                as a side-path (caller then falls back to baking the whole layer)."""
                prepared = []
                for patch in patches or []:
                    if len(patch) == 4:
                        v, _offset, _function, strength = patch
                    else:
                        v, _offset, _function = patch
                        strength = 1.0
                    if not self._dyn_capable(v):
                        return None
                    w = getattr(v, "weights", None)
                    if isinstance(w, (tuple, list)):
                        moved = [(t.to(device="cuda", dtype=torch.float16)
                                  if isinstance(t, torch.Tensor) and t.is_floating_point() else t)
                                 for t in w]
                        v.weights = tuple(moved)
                    # Attributes BypassForwardHook would otherwise set. Only Linear
                    # layers are wrapped here, so no conv metadata is needed.
                    v.multiplier = float(strength)
                    v.is_conv = False
                    v.conv_dim = 0
                    v.kernel_size = (1,)
                    v.in_channels = None
                    v.out_channels = None
                    v.kw_dict = {}
                    prepared.append(v)
                return prepared

            def _apply_dyn_lora(self, x, out):
                """Add each attached LoRA's low-rank contribution to the int4 output.
                h() returns up(down(x)) already scaled by (alpha/rank)*strength."""
                dyn = self._dyn_lora
                if not dyn:
                    return out
                for v in dyn:
                    try:
                        out = out + v.h(x, out)
                    except Exception as e:
                        logging.warning(f"INT4 ConvRot: dynamic LoRA side-path failed: {e}")
                return out

            @staticmethod
            def _exact_key(prefix):
                """Normalized exact key for a module prefix (drops the trailing
                '.' and any container prefix), matching int4_sensitivity.normalize_key
                so auto-selected sets compare exactly rather than by substring."""
                k = prefix[:-1] if prefix.endswith(".") else prefix
                for p in ("diffusion_model.", "model.diffusion_model.", "model.", "transformer."):
                    if k.startswith(p):
                        return k[len(p):]
                return k

            def _should_quantize(self, prefix, w):
                if not Int4ConvRotOps.enable or w is None or w.dim() != 2:
                    return False
                if self.in_features < Int4ConvRotOps.min_in_features:
                    return False
                if _pick_groupsize(self.in_features) is None:
                    return False
                # name heuristic (substring) OR auto-selected bf16 (exact key).
                if any(n in prefix for n in (Int4ConvRotOps.excluded_names or [])):
                    return False
                if self._exact_key(prefix) in Int4ConvRotOps.excluded_exact:
                    return False
                return True

            def _quant_mode(self, prefix):
                """Which precision to quantize this (already-eligible) layer to."""
                return "int8" if self._exact_key(prefix) in Int4ConvRotOps.int8_exact else "int4"

            def _load_prequantized(self, state_dict, prefix, wkey, skey, w):
                """Load an already-quantized ConvRot weight straight from the
                checkpoint (no re-quant). The stored column count tells int4 from
                int8: int4 is packed to K/2, int8 is plain K. Returns True if
                consumed. Handles both the legacy files written by this node and
                ComfyUI native convrot files (per-layer 'comfy_quant' tag, int8
                scale stored as [N, 1])."""
                cols = w.shape[-1]
                if cols == self.in_features:
                    mode = "int8"
                elif cols * 2 == self.in_features:
                    mode = "int4"
                else:
                    return False  # unrecognized layout; fall back to bf16 load
                # Prefer the groupsize recorded in a ComfyUI native tag; otherwise
                # recompute it deterministically from in_features (legacy files).
                gs = None
                qkey = prefix + "comfy_quant"
                qtag = state_dict.get(qkey, None)
                if qtag is not None:
                    try:
                        import json
                        conf = json.loads(bytes(qtag.cpu().numpy().tolist()).decode("utf-8"))
                        fmt = conf.get("format")
                        if fmt not in ("convrot_w4a4", "int8_tensorwise"):
                            return False  # e.g. an fp8 layer the kernel can't run
                        # The int8 path is convrot-only; a plain (unrotated) int8
                        # layer would be mismatched by convrot_w8a8_linear.
                        if fmt == "int8_tensorwise" and not conf.get("convrot", False):
                            return False
                        gs = int(conf.get("convrot_groupsize", 0)) or None
                    except Exception:
                        gs = None
                if gs is None:
                    gs = _pick_groupsize(self.in_features)
                if gs is None:
                    return False
                dev = torch.device("cuda")
                self._q = w.to(dev).contiguous()
                # Scale is per-row [N]; native int8 stores it as [N, 1] -> flatten.
                self._qs = state_dict[skey].to(dev, torch.float32).reshape(-1).contiguous()
                self._qgs = gs
                self._qmode = mode
                self.weight = None
                state_dict.pop(wkey, None)
                state_dict.pop(skey, None)
                state_dict.pop(qkey, None)
                bkey = prefix + "bias"
                if bkey in state_dict:
                    self.bias = nn.Parameter(
                        state_dict.pop(bkey).to(dev, torch.float16), requires_grad=False)
                # Dynamic LoRA over a prequant base: no bf16 weight to bake into, but
                # the fp16 side-path only needs the adapter, so attach it here too.
                if Int4ConvRotOps.lora_dynamic and Int4ConvRotOps.lora_patches:
                    nk = self._normalize_lora_key(wkey)
                    patches = Int4ConvRotOps.lora_patches.get(nk)
                    if patches:
                        prepared = self._prepare_dyn_adapters(patches)
                        if prepared is not None:
                            self._dyn_lora = prepared
                            Int4ConvRotOps._applied_lora.add(nk)
                return True

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                wkey = prefix + "weight"
                skey = prefix + "weight_scale"
                w = state_dict.get(wkey, None)

                # (1) Prequantized: checkpoint already carries packed int4 + scale.
                # (LoRA cannot be baked into already-quantized weights.)
                if w is not None and w.dtype == torch.int8 and skey in state_dict:
                    try:
                        if self._load_prequantized(state_dict, prefix, wkey, skey, w):
                            return
                    except Exception as e:
                        logging.warning(f"INT4 ConvRot: prequant load failed for {prefix}, "
                                        f"falling back to bf16: {e}")

                # LoRA. Two modes:
                #   dynamic: for a quantized layer, keep the low-rank adapter and
                #     add it as a side-path in forward() (base weights stay int4, so
                #     strengths can change without re-quantizing).
                #   bake (default): fold patches into the bf16 weight/bias before
                #     quantizing (or storing) it; written back so path (3) sees them.
                # bf16-kept layers always bake (they have a real weight, exact).
                if Int4ConvRotOps.lora_patches:
                    nk = self._normalize_lora_key(wkey)
                    wpatches = Int4ConvRotOps.lora_patches.get(nk)
                    prepared = None
                    if (Int4ConvRotOps.lora_dynamic and wpatches
                            and w is not None and w.dtype != torch.int8
                            and self._should_quantize(prefix, w)):
                        prepared = self._prepare_dyn_adapters(wpatches)
                    if prepared is not None:
                        self._dyn_lora = prepared
                        Int4ConvRotOps._applied_lora.add(nk)
                    else:
                        if w is not None and w.dtype != torch.int8:
                            wp = self._apply_lora(w, wkey)
                            if wp is not w:
                                w = wp
                                state_dict[wkey] = w
                        bkey = prefix + "bias"
                        b = state_dict.get(bkey, None)
                        if b is not None:
                            bp = self._apply_lora(b, bkey)
                            if bp is not b:
                                state_dict[bkey] = bp

                # (2) On-the-fly quantization of a (possibly LoRA-baked) bf16 weight.
                if self._should_quantize(prefix, w):
                    dev = torch.device("cuda")
                    gs = _pick_groupsize(self.in_features)
                    mode = self._quant_mode(prefix)
                    try:
                        if mode == "int8":
                            from ._int4.int8_eager import quantize_convrot_w8a8_weight
                            qw, ws = quantize_convrot_w8a8_weight(
                                w.to(dev, torch.float16), convrot_groupsize=gs)
                        else:
                            from ._int4 import quantize_convrot_w4a4_weight
                            stoch = (Int4ConvRotOps._STOCHASTIC_SEED
                                     if Int4ConvRotOps.lora_mode == "Stochastic" else 0)
                            qw, ws = quantize_convrot_w4a4_weight(
                                w.to(dev, torch.float16), convrot_groupsize=gs,
                                stochastic_rounding=stoch)
                        self._q, self._qs, self._qgs, self._qmode = qw, ws.float(), gs, mode
                        self.weight = None
                        state_dict.pop(wkey, None)
                        bkey = prefix + "bias"
                        if bkey in state_dict:
                            self.bias = nn.Parameter(
                                state_dict.pop(bkey).to(dev, torch.float16), requires_grad=False)
                        return
                    except Exception as e:
                        logging.warning(f"INT4 ConvRot: fell back to bf16 for {prefix}: {e}")

                # (3) Plain bf16/fp32 load (LoRA-baked weight already in state_dict).
                return super()._load_from_state_dict(
                    state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs)

            def forward(self, x):
                if self._q is None:
                    return self._apply_dyn_lora(x, super().forward(x))
                if self._q.device != x.device:
                    self._q = self._q.to(x.device)
                    self._qs = self._qs.to(x.device)
                bias = self.bias
                if bias is not None and bias.device != x.device:
                    bias = bias.to(x.device)
                if self._qmode == "int8":
                    from ._int4 import convrot_w8a8_linear
                    out = convrot_w8a8_linear(x, self._q, self._qs, bias=bias,
                                              convrot_groupsize=self._qgs)
                else:
                    from ._int4 import convrot_w4a4_linear
                    out = convrot_w4a4_linear(x, self._q, self._qs, bias=bias,
                                              convrot_groupsize=self._qgs)
                return self._apply_dyn_lora(x, out)

else:  # comfy.ops or the vendored int4 backend unavailable
    Int4ConvRotOps = None
