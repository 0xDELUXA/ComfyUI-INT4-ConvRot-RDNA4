"""Pre-LoRA loader: select one or more LoRAs to bake into the weights during
int4 quantization (the loader folds them into bf16 before packing to int4).

LoRA cannot be applied to an already-int4 model at inference (activations are
quantized), so it is baked at load time. Bake a bf16 model + LoRA, then optionally
Save the result to an int4 checkpoint that already contains the LoRA.
"""
import folder_paths


class INT4PreLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name_1": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "hidden": {"prompt": "PROMPT", "id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PRE_LORA",)
    FUNCTION = "load_pre_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Select LoRA(s) to bake into an INT4 ConvRot model at load time."

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    def load_pre_lora(self, **kwargs):
        loras = []
        # ComfyUI strips dynamic inputs not declared in INPUT_TYPES; recover them
        # from the raw prompt dict so extra lora_name_N widgets still work.
        prompt = kwargs.get("prompt", {})
        node_id = kwargs.get("id", None)
        node_inputs = prompt[node_id]["inputs"] if (prompt and node_id and node_id in prompt) else kwargs

        i = 1
        while True:
            name_key = f"lora_name_{i}"
            if name_key not in node_inputs:
                break
            name = node_inputs[name_key]
            strength = round(float(node_inputs.get(f"lora_strength_{i}", 1.0)), 4)
            if name != "None" and strength != 0.0:
                loras.append({"lora_name": name, "lora_strength": strength})
            i += 1
        return (loras,)


NODE_CLASS_MAPPINGS = {"INT4PreLoraLoader": INT4PreLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4PreLoraLoader": "INT4 ConvRot Pre-LoRA"}
