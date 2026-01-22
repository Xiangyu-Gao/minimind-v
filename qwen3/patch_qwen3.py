"""
Monkey patch for Qwen3-VL configuration issue in transformers 5.0.0.dev0
"""
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_module


# Store original __init__
original_init = qwen3_vl_module.Qwen3VLTextModel.__init__


def patched_init(self, config):
    # Add pad_token_id to config if it doesn't exist
    if not hasattr(config, "pad_token_id"):
        config.pad_token_id = (
            config.eos_token_id if hasattr(config, "eos_token_id") else None
        )

    # Call original init
    original_init(self, config)


# Apply the patch
qwen3_vl_module.Qwen3VLTextModel.__init__ = patched_init

print("Qwen3-VL configuration patch applied successfully!")
