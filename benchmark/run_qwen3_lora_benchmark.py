"""
Benchmark evaluation using LoRA fine-tuned Qwen3-VL model.
This uses the LoRA weights to produce reference-style aligned outputs.
"""

import os
import sys
import argparse
import warnings
import torch

__package__ = "evaluator"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Apply Qwen3-VL patch BEFORE importing transformers models
sys.path.insert(0, os.path.dirname(__file__))
import qwen3.patch_qwen3 as patch_qwen3  # noqa

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from benchmark.bench_utils import (
    add_common_args,
    init_translator,
    translate_response,
    load_eval_data,
    collect_prediction,
    export_and_evaluate,
    ProgressTracker,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA Benchmark Evaluation")
    parser.add_argument(
        "--base_model_path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        type=str,
        help="Base model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--lora_path",
        default="./qwen3_lora_output",
        type=str,
        help="Path to LoRA weights",
    )

    # Common benchmark args
    add_common_args(parser)
    parser.set_defaults(max_new_tokens=256)

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-VL LoRA Benchmark Evaluation")
    print("=" * 60)

    # Initialize translator
    translator, backend = init_translator(args.language, backend="deep_translator")

    # Load processor
    print(f"\nLoading processor from {args.base_model_path}...")
    processor = AutoProcessor.from_pretrained(args.base_model_path)

    # Load base model
    print(f"Loading base model from {args.base_model_path}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.merge_and_unload()  # Merge for faster inference
    model.eval()

    print(f"Model loaded successfully on {model.device}")
    print(f"Model dtype: {model.dtype}")

    # Load test data
    json_data, mapping_data = load_eval_data(args.data_path)

    predictions = []
    tracker = ProgressTracker(len(json_data))

    for idx, qa_pair in enumerate(json_data):
        prompt = qa_pair["conversations"][0]["content"]
        image_paths = qa_pair["image"]

        # Prepare image paths
        images = []
        for image_name in image_paths.split(","):
            image_path = os.path.join(args.image_dir, image_name.strip())
            images.append(image_path)

        if args.use_single_image == 1:
            images = images[:1]

        # Construct messages with system prompt for concise output
        content = []
        parts = prompt.split("<image>")

        for i, part in enumerate(parts):
            if i < len(images):
                content.append({"type": "image", "image": images[i]})
            if part.strip():
                content.append({"type": "text", "text": part.strip()})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise, direct answers. Avoid lengthy reasoning or explanations.",
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(args.device)

        tracker.update(idx, image_paths, prompt)

        # Generate
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            generated_ids = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(response)

        # Translate response if language is set to English
        if args.language == "english":
            response = translate_response(response, translator, backend)

        predictions.append(collect_prediction(mapping_data, idx, response))

    # Export and evaluate
    export_and_evaluate(predictions, args.dump_eval_dir, args.data_path, "qwen3vl_lora")


if __name__ == "__main__":
    main()
