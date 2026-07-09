"""INT4 ConvRot-W4A4 for ComfyUI on AMD RDNA4 (gfx12).

On-the-fly int4 quantization of diffusion-model Linear layers, running a native
HIP wide-K int4 WMMA kernel (vendored, self-contained). Companion to the INT8
node, for the int4 (W4A4) regime on large-hidden DiTs (Flux/Krea/Qwen-image).
"""
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .int4_loader import (
        NODE_CLASS_MAPPINGS as _loader_cls,
        NODE_DISPLAY_NAME_MAPPINGS as _loader_names,
    )
    NODE_CLASS_MAPPINGS.update(_loader_cls)
    NODE_DISPLAY_NAME_MAPPINGS.update(_loader_names)
except Exception as e:  # keep ComfyUI booting even if deps are missing
    logging.error(f"INT4 ConvRot: loader node import failed: {e}")

try:
    from .int4_save import (
        NODE_CLASS_MAPPINGS as _save_cls,
        NODE_DISPLAY_NAME_MAPPINGS as _save_names,
    )
    NODE_CLASS_MAPPINGS.update(_save_cls)
    NODE_DISPLAY_NAME_MAPPINGS.update(_save_names)
except Exception as e:
    logging.error(f"INT4 ConvRot: save node import failed: {e}")

try:
    from .int4_lora import (
        NODE_CLASS_MAPPINGS as _lora_cls,
        NODE_DISPLAY_NAME_MAPPINGS as _lora_names,
    )
    NODE_CLASS_MAPPINGS.update(_lora_cls)
    NODE_DISPLAY_NAME_MAPPINGS.update(_lora_names)
except Exception as e:
    logging.error(f"INT4 ConvRot: pre-lora node import failed: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
