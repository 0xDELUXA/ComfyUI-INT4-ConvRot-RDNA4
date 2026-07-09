"""Loader node: load a bf16 diffusion model and quantize it to int4 ConvRot-W4A4
on the fly, running the native HIP int4 kernel on AMD RDNA4 (gfx12)."""
import json
import logging

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_detection
import comfy.sd
import comfy.utils
import folder_paths

from .int4_ops import Int4ConvRotOps, _pick_groupsize, hip_backend_ready
from .int4_sensitivity import (
    format_ranking, normalize_key, rank_layers, select_precision,
)

# Per-model sensitive-layer skip lists (mixed precision): keep embeddings /
# conditioning / modulation / final layers at bf16. int4 (W4A4) is more
# sensitive than int8, so these lists are a safe floor - if a model's quality
# degrades, add more substrings here.
EXCLUDED = {
    "flux2": ['img_in', 'time_in', 'guidance_in', 'txt_in',
              'double_stream_modulation_img', 'double_stream_modulation_txt',
              'single_stream_modulation', 'final_layer'],
    "krea2": ['first', 'last', 'tmlp', 'tproj', 'txtfusion', 'txtmlp'],
    "qwen": ['time_text_embed', 'img_in', 'norm_out', 'proj_out', 'txt_in',
             'mlp.net.2', 'transformer_blocks.59.'],
    "chroma": ['distilled_guidance_layer', 'final_layer', 'img_in', 'txt_in',
               'nerf_image_embedder', 'nerf_blocks', 'nerf_final_layer_conv', '__x0__'],
    "wan": ['patch_embedding', 'text_embedding', 'time_embedding', 'time_projection',
            'head', 'img_emb', 'face_adapter', 'face_encoder', 'motion_encoder',
            'pose_patch_embedding'],
    "z-image": ['cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token',
                'context_refiner', 'final_layer', 'noise_refiner', 'adaLN',
                'x_pad_token', 'layers.0.', 'feed_forward.w2', 'layers.29.'],
    "ltx2": ['adaln', 'embedding', 'patchify', 'to_gate_logits', 'proj_out',
             'model.audio', 'model.video', 'model.av', 'model.patch',
             'model.proj', 'shift'],
    "ernie": ['time', 'x_embedder', 'text_proj', 'adaLN'],
    "anima": ['embed', 'llm', 'adaln'],
    "hidream": ['embed', 'language_model.layers.35.mlp'],
    "boogu": ['embed', 'refine', 'norm_out'],
    "ideogram4": ['embed_image_indicator', 't_embedding', 'proj'],
    "generic-dit": ['embed', 'time', 'txt_in', 'img_in', 'final', 'proj_out',
                    'norm_out', 'adaLN', 'adaln'],
}


def _normalize_key(key):
    if not isinstance(key, str):
        return key
    for p in ("diffusion_model.", "model.diffusion_model.", "model.", "transformer."):
        if key.startswith(p):
            return key[len(p):]
    return key


def _native_unsupported_layers(sd):
    """List (layer_key, format) for ComfyUI native comfy_quant layers the RDNA4
    convrot kernel cannot run: anything other than convrot_w4a4, or int8_tensorwise
    without convrot rotation (fp8, nvfp4, plain int8, etc.). The kernel quantizes
    activations too (W4A4/W8A8 convrot), so these formats would compute wrong."""
    bad = []
    suffix = ".comfy_quant"
    for k, v in sd.items():
        if not isinstance(k, str) or not k.endswith(suffix):
            continue
        try:
            conf = json.loads(bytes(v.cpu().to(torch.uint8).tolist()).decode("utf-8"))
        except Exception:
            continue
        fmt = conf.get("format")
        ok = fmt == "convrot_w4a4" or (fmt == "int8_tensorwise" and conf.get("convrot", False))
        if not ok:
            bad.append((k[:-len(suffix)], fmt))
    return bad


def _build_lora_patches(sd, metadata, loras):
    """Build {normalized_layer_key: [(v, offset, function, strength), ...]} from a
    list of {lora_name, lora_strength}. Uses a meta-device skeleton model to
    discover the LoRA -> layer key map (Int4ConvRotOps.Linear honors
    skeleton_meta_init). Returns {} on any failure (load still proceeds)."""
    grouped = {}
    if not loras:
        return grouped

    unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    m_config = comfy.model_detection.model_config_from_unet(sd, unet_prefix, metadata=metadata)
    if m_config is None and unet_prefix != "":
        m_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
        if m_config is not None:
            unet_prefix = ""
    if m_config is None:
        logging.warning("INT4 ConvRot: could not detect model for LoRA key mapping; skipping LoRA.")
        return grouped

    m_config.custom_operations = Int4ConvRotOps
    Int4ConvRotOps.skeleton_meta_init = True
    try:
        skeleton = m_config.get_model(sd, unet_prefix)
        key_map = comfy.lora.model_lora_keys_unet(skeleton, {})
    finally:
        Int4ConvRotOps.skeleton_meta_init = False
    del skeleton

    for lora in loras:
        name = lora.get("lora_name", "None")
        strength = float(lora.get("lora_strength", 1.0))
        if name == "None" or strength == 0.0:
            continue
        lora_path = folder_paths.get_full_path("loras", name)
        if lora_path is None:
            logging.warning(f"INT4 ConvRot: LoRA '{name}' not found; skipping.")
            continue
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        lora_data = comfy.lora_convert.convert_lora(lora_data)
        patch_dict = comfy.lora.load_lora(lora_data, key_map)
        for k, v in patch_dict.items():
            target_key, offset, function = k, None, None
            if isinstance(k, tuple):
                target_key = k[0]
                if len(k) > 1:
                    offset = k[1]
                if len(k) > 2:
                    function = k[2]
            grouped.setdefault(_normalize_key(target_key), []).append((v, offset, function, strength))
        del lora_data
        _how = "dynamic application" if Int4ConvRotOps.lora_dynamic else "baking"
        logging.info(f"INT4 ConvRot: prepared LoRA '{name}' (strength {strength}) for {_how}.")

    return grouped


class UNetLoaderINT4ConvRot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (list(EXCLUDED.keys()),
                               {"tooltip": "Selects which sensitive layers stay bf16 (mixed precision)."}),
                "min_in_features": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64,
                                            "tooltip": "Skip Linear layers narrower than this (int4 overhead > win)."}),
                "lora_mode": (["None", "Stochastic", "Dynamic"],
                              {"default": "None",
                               "tooltip": "How LoRAs are applied. None = bake (round); Stochastic = "
                                          "bake with stochastic int4 rounding (sometimes closer to "
                                          "bf16+LoRA); Dynamic = keep the LoRA as a runtime side-path "
                                          "on int4 layers (base weights untouched, strengths can "
                                          "change without re-quantizing; slightly slower per step)."}),
            },
            "optional": {
                "pre_lora": ("PRE_LORA", {"tooltip": "LoRA(s) from an INT4 ConvRot Pre-LoRA node, "
                                          "baked into the weights during quantization."}),
                "report_sensitivity": ("BOOLEAN", {"default": False,
                                        "tooltip": "Log a calibration-free per-layer int4 error ranking "
                                                   "(worst-first) before loading. Diagnostic only; does "
                                                   "not change which layers are quantized."}),
                "auto_protect_vram_mb": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64,
                                        "tooltip": "Extra VRAM (MB) to spend keeping the most int4-damaged "
                                                   "layers at full precision, chosen automatically by "
                                                   "value-per-MB. 0 = pure int4 (smallest/fastest, current "
                                                   "behaviour). Higher = better quality, more VRAM. Adds a "
                                                   "one-time scoring pass at load."}),
                "scoring_mode": (["weight (fast)", "output (activation-aware)"],
                                 {"default": "weight (fast)",
                                  "tooltip": "How layers are scored for report/auto-protect. 'weight' = "
                                             "fast weight round-trip error. 'output' = measures the full "
                                             "int4/int8 output drift on probe activations (catches "
                                             "activation-quant sensitivity too); slower but more accurate."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load a bf16 diffusion model and quantize to int4 ConvRot-W4A4 (AMD RDNA4 HIP)."

    def load_unet(self, unet_name, model_type, min_in_features, lora_mode="None", pre_lora=None,
                  report_sensitivity=False, auto_protect_vram_mb=0, scoring_mode="weight (fast)"):
        if Int4ConvRotOps is None:
            raise RuntimeError("INT4 ConvRot: comfy.ops or the vendored int4 backend is unavailable.")
        if not hip_backend_ready():
            raise RuntimeError("INT4 ConvRot: HIP backend not ready (need gfx12 ROCm + built kernel). "
                               "First load compiles the kernel via hipcc - ensure ROCm/hipcc is available.")

        loras = pre_lora if isinstance(pre_lora, list) else ([pre_lora] if pre_lora else [])

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        # A prequantized checkpoint carries packed int4 weights already, so
        # on-the-fly quantization is disabled (the per-layer prequant detection in
        # Int4ConvRotOps.Linear loads them directly, and the intentionally-bf16
        # layers must stay bf16 rather than being re-quantized). Two markers are
        # recognised: the header tag written by this node, and ComfyUI's per-layer
        # "comfy_quant" tags, so a natively quantized convrot file (or one saved in
        # native format) also loads here and runs on the RDNA4 kernel.
        has_marker = isinstance(metadata, dict) and metadata.get("convrot_w4a4") is not None
        has_native = any(k.endswith(".comfy_quant") for k in sd)
        is_prequant = has_marker or has_native

        # A native file may carry layers in formats this kernel cannot run (fp8,
        # nvfp4, non-convrot int8). Refuse up front with a clear message instead of
        # loading them wrong (they would silently produce garbage output).
        if has_native:
            bad = _native_unsupported_layers(sd)
            if bad:
                fmts = ", ".join(sorted(set(str(f) for _, f in bad)))
                raise RuntimeError(
                    f"INT4 ConvRot: {unet_name} has {len(bad)} layer(s) in quantization "
                    f"format(s) the RDNA4 convrot kernel cannot run ({fmts}); e.g. "
                    f"{bad[0][0]}. Load this file with ComfyUI's normal Load Diffusion "
                    f"Model node, or re-quantize from a bf16 source with this node.")

        # Reset LoRA state each load so a previous run's LoRAs don't stick.
        Int4ConvRotOps.lora_mode = str(lora_mode)
        Int4ConvRotOps.lora_dynamic = (str(lora_mode) == "Dynamic")
        Int4ConvRotOps._applied_lora = set()
        Int4ConvRotOps.lora_patches = {}

        # Reset per-layer auto-selection each load (prevents a previous model's
        # int8/bf16 picks from leaking into this one).
        Int4ConvRotOps.excluded_exact = set()
        Int4ConvRotOps.int8_exact = set()

        if is_prequant:
            Int4ConvRotOps.enable = False
            Int4ConvRotOps.excluded_names = []
            if loras:
                # Baking needs the bf16 weight, which a prequant file no longer has.
                # Dynamic mode does not: it adds the LoRA as an fp16 side-path over
                # the packed int4 base, so it works here (and skips re-quantization
                # entirely, so changing lora_strength only re-attaches the side-path).
                if Int4ConvRotOps.lora_dynamic:
                    try:
                        Int4ConvRotOps.lora_patches = _build_lora_patches(sd, metadata, loras)
                    except Exception as e:
                        logging.warning(f"INT4 ConvRot: LoRA preparation failed ({e}); "
                                        f"loading without LoRA.")
                        Int4ConvRotOps.lora_patches = {}
                else:
                    logging.warning("INT4 ConvRot: baked LoRA is not supported on a prequantized "
                                    "int4 model (weights are already packed). Set lora_mode to "
                                    "Dynamic to apply a LoRA as a runtime side-path, or load a bf16 "
                                    "source to bake. Ignoring LoRA.")
            _fmt = (f"ComfyUI native convrot" if has_native and not has_marker
                    else f"format v{metadata.get('convrot_w4a4')}")
            logging.info(f"INT4 ConvRot: loading PREQUANTIZED {unet_name} "
                         f"({_fmt})"
                         f"{f', applying {len(Int4ConvRotOps.lora_patches)} LoRA layer(s) dynamically' if Int4ConvRotOps.lora_patches else ''}.")
        else:
            Int4ConvRotOps.enable = True
            Int4ConvRotOps.excluded_names = EXCLUDED.get(model_type, EXCLUDED["generic-dit"])
            Int4ConvRotOps.min_in_features = int(min_in_features)
            if loras:
                try:
                    Int4ConvRotOps.lora_patches = _build_lora_patches(sd, metadata, loras)
                except Exception as e:
                    logging.warning(f"INT4 ConvRot: LoRA preparation failed ({e}); loading without LoRA.")
                    Int4ConvRotOps.lora_patches = {}
            _lora_verb = "applying" if Int4ConvRotOps.lora_dynamic else "baking"
            logging.info(f"INT4 ConvRot: loading {unet_name} (excluding "
                         f"{len(Int4ConvRotOps.excluded_names)} layer groups), "
                         f"quantizing to int4 W4A4 on the fly"
                         f"{f', {_lora_verb} {len(Int4ConvRotOps.lora_patches)} LoRA layer(s) [{lora_mode}]' if Int4ConvRotOps.lora_patches else ''}.")

            # Optional one-time scoring pass, shared by the diagnostic report and
            # the automatic int8/bf16 selection. Best-effort: never breaks the load.
            budget_mb = int(auto_protect_vram_mb)
            if report_sensitivity or budget_mb > 0:
                try:
                    base_excluded = list(Int4ConvRotOps.excluded_names)
                    mode = "output" if str(scoring_mode).startswith("output") else "weight"
                    # Only measure int8 error when the selector will actually use it.
                    records = rank_layers(sd, int(min_in_features), _pick_groupsize,
                                          excluded_names=base_excluded,
                                          include_int8=budget_mb > 0, scoring_mode=mode)
                    if report_sensitivity:
                        logging.info(format_ranking(records))
                    if budget_mb > 0:
                        bf16, int8, spent = select_precision(records, budget_mb * 1024 * 1024)
                        if bf16 or int8:
                            bf16_keys = [normalize_key(r["key"]) for r in bf16]
                            int8_keys = [normalize_key(r["key"]) for r in int8]
                            Int4ConvRotOps.excluded_exact = set(bf16_keys)
                            Int4ConvRotOps.int8_exact = set(int8_keys)
                            logging.info(
                                f"INT4 ConvRot: auto-protect spent {spent / 1024 / 1024:.0f} MB of a "
                                f"{budget_mb} MB budget -> {len(int8_keys)} layer(s) to int8, "
                                f"{len(bf16_keys)} layer(s) to bf16 (rest int4). Most damaged int8: "
                                f"{', '.join(int8_keys[:4]) or '-'}; bf16: {', '.join(bf16_keys[:4]) or '-'}.")
                        else:
                            logging.info(f"INT4 ConvRot: auto-protect budget {budget_mb} MB "
                                         f"selected no upgrades (all layers stay int4).")
                except Exception as e:
                    logging.warning(f"INT4 ConvRot: sensitivity/auto-protect pass failed ({e}); "
                                    f"using name heuristic only.")

        model_options = {"custom_operations": Int4ConvRotOps}
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options=model_options, metadata=metadata)

        # Report unmatched LoRA keys (helps debugging).
        if Int4ConvRotOps.lora_patches:
            unmatched = set(Int4ConvRotOps.lora_patches.keys()) - Int4ConvRotOps._applied_lora
            if unmatched:
                logging.warning(f"INT4 ConvRot: {len(unmatched)} LoRA key(s) did not match any "
                                f"layer, e.g. {sorted(unmatched)[:3]}")
        # Clear patches so they don't leak into a later load of a different model.
        Int4ConvRotOps.lora_patches = {}

        # Stash source metadata so the Save node can round-trip it. ModelPatcher
        # attributes are dropped by clone(), but the inner model object is shared
        # by reference, so attach it there too.
        model._safetensors_metadata = metadata
        try:
            if model.model is not None:
                model.model._int4_source_metadata = metadata
        except Exception:
            pass
        return (model,)


NODE_CLASS_MAPPINGS = {"UNetLoaderINT4ConvRot": UNetLoaderINT4ConvRot}
NODE_DISPLAY_NAME_MAPPINGS = {"UNetLoaderINT4ConvRot": "Load Diffusion Model INT4 ConvRot (AMD)"}
