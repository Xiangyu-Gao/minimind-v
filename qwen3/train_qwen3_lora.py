"""
LoRA fine-tuning script for Qwen3-VL to align output style with reference SFT dataset.
This performs reference-style distillation to produce concise, caption-style outputs.
"""

import os
import sys
import argparse
import warnings
import json
import torch
from torch.utils.data import Dataset

__package__ = "benchmark"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Apply Qwen3-VL patch BEFORE importing transformers models
sys.path.insert(0, os.path.dirname(__file__))
import qwen3.patch_qwen3 as patch_qwen3  # noqa

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")


class SFTVisionDataset(Dataset):
    """Dataset for Qwen3-VL SFT with vision inputs."""

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor: AutoProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.image_dir = image_dir
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, path: str):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def _build_messages(self, conversations, image_paths):
        """Convert conversations to Qwen3-VL message format."""
        messages = []

        # Add system message for concise output style
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise, direct answers. Avoid lengthy reasoning or explanations.",
            }
        )

        for conv in conversations:
            role = conv["role"]
            content = conv["content"]

            if role == "user":
                # Build content with images
                content_parts = []
                parts = content.split("<image>")

                img_idx = 0
                for i, part in enumerate(parts):
                    # Add image before text (if we have images left)
                    if i > 0 and img_idx < len(image_paths):
                        content_parts.append(
                            {"type": "image", "image": image_paths[img_idx]}
                        )
                        img_idx += 1
                    if part.strip():
                        content_parts.append({"type": "text", "text": part.strip()})

                # Handle case where <image> is at the start
                if content.startswith("<image>") and image_paths:
                    content_parts.insert(0, {"type": "image", "image": image_paths[0]})
                    if img_idx == 0:
                        img_idx = 1

                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": "assistant", "content": content})

        return messages

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]
        image_names = item.get("image", "")

        # Get image paths
        image_paths = []
        if image_names:
            for img_name in image_names.split(","):
                img_path = os.path.join(self.image_dir, img_name.strip())
                if os.path.exists(img_path):
                    image_paths.append(img_path)

        # Build messages
        messages = self._build_messages(conversations, image_paths)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze batch dimension for text inputs only
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Keep vision inputs as-is (they have shape [num_images, ...])
        # pixel_values: [num_patches, channels] or [num_images, channels, h, w]
        # image_grid_thw: [num_images, 3] for (temporal, height, width)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Debug: print shapes on first sample
        if idx == 0:
            print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
            if pixel_values is not None:
                print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
            if image_grid_thw is not None:
                print(f"[DEBUG] image_grid_thw shape: {image_grid_thw.shape}")

        # Create labels (same as input_ids for causal LM, -100 for padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Don't squeeze vision tensors - they need to maintain [num_images, ...] shape
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw

        return result


def collate_fn(batch):
    """Custom collate function for vision inputs.

    For Qwen3-VL, vision tensors need to be concatenated (not stacked)
    since each sample may have different numbers of images/patches.
    """
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Concatenate vision tensors along dim=0 (not stack)
    # This combines all images from the batch into a single tensor
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        pixel_values = torch.cat([x["pixel_values"] for x in batch], dim=0)
        result["pixel_values"] = pixel_values

    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        image_grid_thw = torch.cat([x["image_grid_thw"] for x in batch], dim=0)
        result["image_grid_thw"] = image_grid_thw

    return result


def print_trainable_parameters(model):
    """Print the number of trainable parameters."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL LoRA Fine-tuning for Reference-Style Distillation"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        type=str,
        help="Base model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output_dir",
        default="./qwen3_lora_output",
        type=str,
        help="Output directory for LoRA weights",
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/LingoQA_train.jsonl",
        help="Training data path (JSONL format)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./dataset/sft_images_v1",
        help="Training image directory",
    )
    parser.add_argument(
        "--max_length",
        default=640,
        type=int,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_samples",
        default=0,
        type=int,
        help="Limit dataset size (0 = use all data)",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        default=64,
        type=int,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        default=128,
        type=int,
        help="LoRA alpha (scaling factor)",
    )
    parser.add_argument(
        "--lora_dropout",
        default=0.05,
        type=float,
        help="LoRA dropout",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=8,
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.03,
        type=float,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--save_steps",
        default=500,
        type=int,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        default=10,
        type=int,
        help="Log every N steps",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--max_pixels",
        default=512,
        type=int,
        help="Max image pixels (N * 28 * 28). Lower = faster. Default 512, original 1280",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-VL LoRA Fine-tuning for Reference-Style Distillation")
    print("=" * 60)

    # Load processor with reduced image resolution for faster training
    # Default Qwen3-VL uses max_pixels=1280*28*28, we reduce it significantly
    print(f"\nLoading processor from {args.model_path}...")
    print(f"Image resolution: max_pixels={args.max_pixels}*28*28 (original=1280*28*28)")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels * 28 * 28,
    )

    # Load model
    print(f"Loading model from {args.model_path}...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Enable Flash Attention 2 for faster training (requires flash-attn package)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, attn_implementation="flash_attention_2", **model_kwargs
        )
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention 2 not available ({e}), using default attention")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, **model_kwargs
        )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    train_dataset = SFTVisionDataset(
        data_path=args.data_path,
        image_dir=args.image_dir,
        processor=processor,
        max_length=args.max_length,
    )

    # Limit dataset size if specified
    if args.max_samples > 0 and args.max_samples < len(train_dataset):
        from torch.utils.data import Subset

        train_dataset = Subset(train_dataset, range(args.max_samples))
        print(
            f"Dataset limited to {args.max_samples} samples (original: {len(train_dataset.dataset)})"
        )
    else:
        print(f"Dataset size: {len(train_dataset)} samples")

    # Training arguments - optimized for speed
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",  # Fused optimizer is faster
        lr_scheduler_type="cosine",
        torch_compile=False,  # Set True if PyTorch 2.0+ for extra speed (may have issues)
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the final LoRA weights
    print(f"\nSaving LoRA weights to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print(f"LoRA weights saved to: {args.output_dir}")
    print("\nTo use the fine-tuned model:")
    print("  from peft import PeftModel")
    print(f'  model = PeftModel.from_pretrained(base_model, "{args.output_dir}")')


if __name__ == "__main__":
    main()
