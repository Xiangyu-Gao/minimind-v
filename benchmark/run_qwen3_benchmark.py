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
    parser = argparse.ArgumentParser(description="Qwen3-VL Benchmark Evaluation")
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        type=str,
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--temperature",
        default=0.65,
        type=float,
        help="Generation temperature (0-1, higher = more random)",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="Nucleus sampling threshold (0-1)"
    )

    # Common benchmark args
    add_common_args(parser)

    args = parser.parse_args()

    # Initialize translator
    translator, backend = init_translator(args.language, backend="deep_translator")

    print(f"Loading Qwen3-VL model from {args.model_path}...")

    # Load processor first
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load model with proper configuration
    print("Loading model with bfloat16 precision...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print(f"Model loaded successfully on {model.device}")
    print(f"Model dtype: {model.dtype}")

    # Load test data
    json_data, mapping_data = load_eval_data(args.data_path)

    # Iterate through each QA pair and generate responses
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

        # Use only first image if single_image mode
        if args.use_single_image == 1:
            images = images[:1]

        # Construct messages for Qwen3-VL format
        content = []
        parts = prompt.split("<image>")

        for i, part in enumerate(parts):
            if i < len(images):
                content.append({"type": "image", "image": images[i]})
            if part.strip():
                content.append({"type": "text", "text": part.strip()})

        messages = [
            {
                "role": "user",
                "content": content,
            }
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

        # Generate (greedy decoding is faster than sampling)
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

        # Decode only the generated tokens
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
    export_and_evaluate(predictions, args.dump_eval_dir, args.data_path, "qwen3vl")


if __name__ == "__main__":
    main()
