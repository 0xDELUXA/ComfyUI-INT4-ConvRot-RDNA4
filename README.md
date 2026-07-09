# ComfyUI-INT4-ConvRot-RDNA4

Native int4 ConvRot-W4A4 quantization for ComfyUI, built specifically for AMD RDNA4 (gfx12 - Radeon RX 9000 series).

Quantizes a diffusion model's Linear layers to 4-bit and runs them through a native HIP wide-K `wmma_i32_16x16x32_iu4` kernel - for the W4A4 regime on large-hidden DiTs (Flux / Krea / Qwen-Image / Z-Image ...).

> RDNA4 / gfx12 only. The int4 matmul uses the `wmma_i32_16x16x32_iu4` instruction introduced on RDNA4 (gfx12). It does not run on RDNA3 or earlier - on a non-gfx12 device the node reports its backend as unavailable and refuses to load. This is not a general ROCm/AMD node.

Inspired by Comfy Org's comfy-kitchen (the int4 ConvRot-W4A4 layout, from which the `_int4/` backend is derived) and by the [ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM) node.

- Self-contained - no external `comfy_kitchen` install; the int4 backend is vendored.
- Source-only - the HIP kernel is compiled on first use from `int4_gemm.hip` via `hipcc`. No prebuilt binaries are shipped.
- Convert once, reuse - quantize a bf16 model on the fly, or save it to an int4 `.safetensors` (~3.5x smaller) and reload it directly (no re-quant, and it no longer needs the full bf16 model to fit in VRAM).

## Requirements

- AMD RDNA4 GPU (gfx12).
- A ROCm build of PyTorch, and `hipcc` available. If ROCm was installed via pip (the `rocm-sdk` / `_rocm_sdk_devel` wheels), the node finds `hipcc` inside the venv automatically; otherwise put `hipcc` on `PATH`.
- The first model load compiles the kernel (one-time, a few seconds).

No pip dependencies beyond what ComfyUI + ROCm PyTorch already provide.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/0xDELUXA/ComfyUI-INT4-ConvRot-RDNA4
```

Restart ComfyUI. On the first int4 sample you'll see a one-time `building: hipcc ... int4_gemm.hip` line in the console.

## When int4 is worth it

int4 W4A4 wins on large-hidden DiTs (Flux-class, hidden >= ~3072) and at higher token counts / resolutions. On small UNets (e.g. SD1.5) or at tiny resolutions the per-layer rotation overhead outweighs the 4-bit GEMM saving and int4 can be slower than bf16/int8 - use it where it pays off.

The Hadamard ConvRot rotation is mandatory for W4A4 quality (it is what makes 4-bit activations viable) - it is always on, not a toggle.

## Nodes

### Load Diffusion Model INT4 ConvRot (AMD)
Drop-in replacement for the standard "Load Diffusion Model" node.

| Input | Meaning |
|---|---|
| `unet_name` | A bf16/fp16 model in `models/diffusion_models/`, a file previously saved by this node, or a ComfyUI native convrot int4/int8 file (all auto-detected and loaded prequantized). |
| `model_type` | Selects the mixed-precision skip list (which sensitive layers stay bf16). |
| `min_in_features` | Skip Linear layers narrower than this (default 512 - below it, int4 overhead > win). Ignored when loading a prequantized file. |
| `lora_mode` | How a Pre-LoRA is applied: None (bake, round), Stochastic (bake, stochastic rounding), or Dynamic (keep as a runtime side-path, base weights stay int4). Only matters when a Pre-LoRA node is attached (see below). |
| `pre_lora` (optional) | LoRA(s) from an INT4 ConvRot Pre-LoRA node, baked into the weights during quantization. |
| `report_sensitivity` (optional) | Log a per-layer int4 error ranking before loading. Diagnostic only; see Tuning quality. |
| `auto_protect_vram_mb` (optional) | VRAM budget (MB) for automatic int4/int8/bf16 mixed precision. 0 (default) = pure int4. See Tuning quality. |
| `scoring_mode` (optional) | How layers are scored for the report and auto-protect: weight (fast) or output (activation-aware). See Tuning quality. |

Output: `MODEL` --> wire into your usual sampler.

### Save Int4 ConvRot Model (AMD)
Takes the `MODEL` from the loader (on-the-fly quantized) and writes an int4 ConvRot `.safetensors` to `output/int4_convrot/`. Move that file into `models/diffusion_models/` and load it with the node above for fast, low-VRAM reloads. (Disconnect this node for normal generation - it's a one-time convert.)

- `save_format`: native (default) or legacy. native writes ComfyUI's per-layer comfy_quant tags, so the same file also loads in the normal Load Diffusion Model node (it runs there on ComfyUI's own convrot kernel, which on RDNA4 falls back to a slow eager path - for speed on RDNA4 load it with this node). legacy writes only this node's header markers. The packed weights are byte-identical between the two, and both reload in the INT4 ConvRot loader.
- The loader also reads a model quantized natively by ComfyUI (convrot int4 / int8), so you can run those on the RDNA4 kernel. A file containing layers in other formats (fp8, nvfp4, non-convrot int8) is rejected up front with a clear error naming the offending layers - load such a file with the normal Load Diffusion Model node instead.

### INT4 ConvRot Pre-LoRA (AMD)
Select one or more LoRAs and wire the `PRE_LORA` output into the loader's `pre_lora` input. The loader's `lora_mode` decides how the LoRA is applied: baked into the bf16 weight before int4 packing (None / Stochastic), or run as a separate side-path at inference (Dynamic). Because activations are also quantized (W4A4), a LoRA cannot be merged into already-packed int4 weights - baking needs the bf16 source, so on a prequantized file only Dynamic works.

- Baking (None / Stochastic) works only when quantizing on the fly (a bf16 source). To ship an int4 model with a LoRA baked in: load bf16 + Pre-LoRA with None or Stochastic, then Save - the saved int4 checkpoint contains the LoRA.
- `lora_mode` on the loader: None (normal rounding) or Stochastic (stochastic int4 rounding - sometimes closer to the bf16+LoRA baseline).
- Dynamic mode is an alternative to baking: the LoRA runs as a small low-rank side-path added on top of the int4 output at inference (the base int4 weights are left untouched). It costs a little speed per step (proportional to LoRA rank). Two benefits: the LoRA math stays in fp16 so it is not subject to int4 rounding (can be closer to the bf16+LoRA baseline than baking), and it works on a prequantized int4 file too - where changing lora_strength_N only re-attaches the side-path instead of re-quantizing. On a bf16 source the base is still re-quantized on every load, so the strength benefit only shows on a prequantized base. Standard LoRAs only; DoRA falls back to baking, and Save does not write a dynamic LoRA into the file.
- Extra `lora_name_N` / `lora_strength_N` widgets on the Pre-LoRA node stack multiple LoRAs.

## Sampler settings

Nothing int4-specific - use the model's normal settings. For distilled models (e.g. Flux.2 Klein) that means CFG 1, euler, ~4 steps.

## On-disk format

A saved file is an ordinary diffusion-model `.safetensors` where quantized layers store `<key>.weight` plus `<key>.weight_scale` (per-row absmax scale); all other layers stay bf16/fp32. A layer's precision is recoverable from its stored weight shape:

- int4: `<key>.weight` is packed int4 (`int8 [N, K/2]`, two signed nibbles per byte).
- int8: `<key>.weight` is plain `int8 [N, K]`.

The Save node's `save_format` picks the container; the packed weights are byte-identical either way:

- native (default): each quantized layer also gets a `<key>.comfy_quant` tag (a small JSON blob naming the format and ConvRot groupsize), matching ComfyUI's `convrot_w4a4` / `int8_tensorwise` layout, so the file also loads in the normal Load Diffusion Model node. Scales are `f32 [N]` for int4 and `f32 [N, 1]` for int8. The groupsize is read from the tag on load.
- legacy: the header carries `convrot_w4a4 = "2"` and a `convrot_precision` JSON map (`{layer_key: "int4"|"int8"}`); scales are `f32 [N]`. int4 vs int8 is decided by the column count on load (v1 all-int4 files still load), and the ConvRot groupsize is recomputed deterministically from `in_features`.

## Tuning quality

If a model degrades in int4, add more layer-name substrings to its list in `int4_loader.py::EXCLUDED` (embeddings, conditioning/time projections, modulation and final layers are the sensitive ones), or raise `min_in_features`.

To decide which substrings to add, enable the loader's `report_sensitivity` toggle. Before loading, it logs a calibration-free per-layer ranking: for every quantizable Linear it measures the int4 ConvRot round-trip error `||W - dequant(quant(W))|| / ||W||` (data-free, no calibration inputs) and prints the layers worst-first, flagging which ones the current name heuristic already keeps at bf16. Layers ranking high but not flagged are the ones most worth protecting. This is diagnostic only - it does not change which layers are quantized.

### Automatic mixed precision (`auto_protect_vram_mb`)

Instead of hand-tuning the name lists, set `auto_protect_vram_mb` on the loader to a VRAM budget (MB). The loader scores every layer (same calibration-free metric as above, for both int4 and int8) and automatically promotes the most int4-damaged layers up the precision ladder - int4 to int8 to bf16 - spending the budget by value per MB (error recovered divided by extra bytes). Because int8 is half the size of bf16 but far more accurate than int4, the selector uses it to stretch the budget: a tight budget buys many layers at int8 rather than a few at bf16; a large budget pushes everything to bf16. `0` (default) means pure int4 and adds no overhead - current behaviour is unchanged. It is purely additive on top of the `EXCLUDED` name list (it only ever raises precision), so it can only improve quality, at the cost of up to about the budget in extra VRAM and a one-time scoring pass at load. A saved model bakes the resulting int4/int8/bf16 mix in, so the budget only matters when quantizing on the fly.

int8 layers run fully on native RDNA4 kernels (both built from the same `int4_gemm.hip`): the matmul on `wmma_i32_16x16x16_iu8`, and the activation rotation + int8 quantize on a fused kernel. They are fast - roughly half int4's throughput (twice the bytes), and about 2x faster than the pure-torch reference. If a kernel is unavailable for a shape it falls back to the eager path automatically.

`scoring_mode` controls how layers are ranked for both the report and auto-protect:
- `weight (fast)` - the calibration-free weight round-trip error (default). Cheap.
- `output (activation-aware)` - runs the real int4/int8 pipeline on probe activations and measures the layer's output drift, so it also catches activation-quantization sensitivity that the weight-only metric misses. Slower (it does probe GEMMs per layer) but more faithful to the real end-to-end error.

## Status & limitations

- LoRA comes from the Pre-LoRA node and is either baked at load time (None / Stochastic) or run as a runtime side-path (Dynamic). The standard ComfyUI Load LoRA node still does not work here: its patches target the int4 weights, which live outside ComfyUI's managed Parameter system and get dropped before inference. Use the Pre-LoRA node instead.
- On-the-fly conversion quantizes layer-by-layer, so the full bf16 model does not need to fit in VRAM - but it must fit in system RAM (it's read from disk to CPU, then each layer is quantized on the GPU transiently).
- The resulting int4 weights stay resident on the GPU - ComfyUI can't stream/offload them (they live outside its managed Parameter system), so a model whose int4 weights alone exceed VRAM won't fit. Full offload streaming isn't implemented yet.
- Text encoders are not quantized by this node (DiT only).

> The WMMA kernel development lives in [amd/hip-backend](https://github.com/0xDELUXA/comfy-kitchen_win-rocm/tree/amd/hip-backend) with the upstream PR available at https://github.com/Comfy-Org/comfy-kitchen/pull/74. For the best performance and overall quality of life, it is recommended to use that implementation alongside native nodes.