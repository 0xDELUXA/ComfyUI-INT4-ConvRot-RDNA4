"""Save node: serialize an int4 ConvRot-W4A4 quantized model to a .safetensors
that this node's loader can reload directly (no re-quantization, ~4x smaller,
and no "full bf16 must fit in VRAM" requirement on reload).

On-disk format (v2), keyed by the model's native diffusion-model key namespace:
  int4 Linear:  <key>.weight        int8  [N, K//2]  (two signed int4 nibbles/byte)
                <key>.weight_scale  f32   [N]         (per-row absmax scale)
  int8 Linear:  <key>.weight        int8  [N, K]      (plain signed int8)
                <key>.weight_scale  f32   [N]         (per-row absmax scale)
  everything else:  plain bf16/fp32 weights, norms, biases (unchanged)
Header metadata carries the format marker "convrot_w4a4"=version and, for v2, a
"convrot_precision" JSON map {layer_key: "int4"|"int8"} for transparency. On load
int4 vs int8 is recovered from the stored column count (K//2 vs K), so v1 files
(all-int4, no map) still load correctly. The ConvRot groupsize is not stored: it
is recomputed deterministically from in_features on load (pick_groupsize).
"""
import json
import logging
import os

import torch

import comfy.model_management
import comfy.utils
import folder_paths

FORMAT_TAG = "convrot_w4a4"
FORMAT_VERSION = "2"
PRECISION_TAG = "convrot_precision"


def _source_metadata(model):
    """Recover the original safetensors metadata stashed by the loader."""
    meta = getattr(model, "_safetensors_metadata", None)
    if isinstance(meta, dict) and meta:
        return meta
    inner = getattr(model, "model", None)
    inner_meta = getattr(inner, "_int4_source_metadata", None)
    if isinstance(inner_meta, dict) and inner_meta:
        return inner_meta
    return {}


def _diffusion_model(model):
    """The DiT submodule whose key namespace matches the source unet file."""
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    return dm if dm is not None else inner


class INT4ConvRotModelSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "int4_convrot/INT4_ConvRot"}),
                "save_format": (["native", "legacy"],
                                {"default": "native",
                                 "tooltip": "native = ComfyUI convrot_w4a4 format (per-layer "
                                            "comfy_quant tags); the file also loads in the normal "
                                            "Load Diffusion Model node. legacy = this node's own "
                                            "header-marker format. Both reload in the INT4 ConvRot "
                                            "loader; the packed weights are byte-identical."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = ("Save an int4 ConvRot-W4A4 quantized model to safetensors. "
                   "Reload it with the INT4 ConvRot loader (prequantized mode).")

    def save(self, model, filename_prefix, save_format="native"):
        native = str(save_format) != "legacy"
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        out_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")

        # Dynamic-VRAM management may have the bf16 params offloaded/staged; force
        # a full on-device load so every parameter is a real, readable tensor.
        # (Packed int4 weights live in module._q, a plain attr that is never
        # offloaded, so they are always readable regardless of VRAM state.)
        try:
            comfy.model_management.load_models_gpu([model], force_full_load=True)
        except Exception as e:
            logging.warning(f"INT4 Save: force_full_load failed ({e}); saving best-effort.")

        dm = _diffusion_model(model)
        if dm is None:
            raise RuntimeError("INT4 Save: could not locate the diffusion model on the patcher.")

        sd = {}
        # Base params/buffers: bf16/excluded weights, norms, biases. Quantized
        # layers have weight=None so their .weight key is absent here.
        for k, v in dm.state_dict().items():
            if isinstance(v, torch.Tensor):
                sd[k] = v.detach().to("cpu").contiguous()

        # Inject packed int4 / plain int8 weights + per-row scales for quantized
        # layers. In native mode also write ComfyUI's per-layer comfy_quant tag so
        # the file loads in the normal Load Diffusion Model node; in legacy mode
        # record each layer's precision in the header map instead.
        precision = {}
        n_int4 = n_int8 = n_dyn = 0
        for name, module in dm.named_modules():
            if getattr(module, "_dyn_lora", None):
                n_dyn += 1
            q = getattr(module, "_q", None)
            if q is None:
                continue
            mode = getattr(module, "_qmode", None) or "int4"
            gs = int(getattr(module, "_qgs", None) or 256)
            scale = module._qs.detach().float().to("cpu").contiguous()
            sd[name + ".weight"] = q.detach().to("cpu").contiguous()
            if native:
                if mode == "int8":
                    # ComfyUI's int8_tensorwise+convrot stores the scale as [N, 1].
                    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
                    scale = scale.reshape(-1, 1).contiguous()
                else:
                    conf = {"format": "convrot_w4a4", "convrot_groupsize": gs}
                sd[name + ".comfy_quant"] = torch.tensor(
                    list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
            sd[name + ".weight_scale"] = scale
            precision[name] = mode
            if mode == "int8":
                n_int8 += 1
            else:
                n_int4 += 1

        if n_int4 + n_int8 == 0:
            logging.warning("INT4 Save: no quantized layers found. Was the model loaded "
                            "with the INT4 ConvRot loader (on-the-fly quantization)?")

        if n_dyn:
            logging.warning(f"INT4 Save: {n_dyn} layer(s) carry a DYNAMIC LoRA side-path that is "
                            f"NOT written to disk. The saved file contains the base weights only. "
                            f"To ship a LoRA baked in, reload with lora_mode None or Stochastic "
                            f"(bake) instead of Dynamic, then Save.")

        # Header metadata: preserve source (str->str only). In legacy mode add this
        # node's format markers; native mode needs none (precision lives in the
        # per-layer comfy_quant tags), keeping the file cleanly ComfyUI-native.
        metadata = {}
        for kk, vv in _source_metadata(model).items():
            if isinstance(kk, str) and isinstance(vv, str):
                metadata[kk] = vv
        if not native:
            metadata[FORMAT_TAG] = FORMAT_VERSION
            metadata[PRECISION_TAG] = json.dumps(precision)

        os.makedirs(full_output_folder, exist_ok=True)
        comfy.utils.save_torch_file(sd, out_path, metadata=metadata)
        logging.info(f"INT4 ConvRot: saved {n_int4} int4 + {n_int8} int8 layer(s) + bf16 "
                     f"remainder ({len(sd)} tensors, {save_format} format) -> {out_path}"
                     f"{' [loads in the normal Load Diffusion Model node too]' if native else ''}")
        return {}


NODE_CLASS_MAPPINGS = {"INT4ConvRotModelSave": INT4ConvRotModelSave}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4ConvRotModelSave": "Save Int4 ConvRot Model (AMD)"}
