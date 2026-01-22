import os
import sys

__package__ = "evaluator"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import warnings
import torch

from PIL import Image
from trainer.trainer_utils import setup_seed
from model.model_vlm import MiniMindVLM, VLMConfig
from benchmark.bench_utils import (
    add_common_args,
    init_translator,
    translate_response,
    load_eval_data,
    collect_prediction,
    export_and_evaluate,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

warnings.filterwarnings("ignore")


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if "model" in args.load_from:
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = f"../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        model = MiniMindVLM(
            VLMConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
            ),
            vision_model_path="../model/vision_model/clip-vit-base-patch16",
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if "mask" not in k}, strict=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model(
            "../model/vision_model/clip-vit-base-patch16"
        )

    print(
        f"VLM模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M(illion)"
    )
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def load_image(image_paths, image_base_dir, preprocess, device):
    """ "Load multiple images and return a tensor stack."""
    setup_seed(2026)  # or setup_seed(random.randint(1, 10000))

    image_tensors = []
    for image_name in image_paths.split(","):
        image_path = os.path.join(image_base_dir, image_name.strip())
        image = Image.open(image_path).convert("RGB")
        pixel_values = (
            MiniMindVLM.image2tensor(image, preprocess).to(device).unsqueeze(0)
        )
        image_tensors.append(pixel_values)

    return torch.stack(image_tensors, dim=0)


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V Chat")
    parser.add_argument(
        "--load_from",
        default="../model",
        type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument(
        "--weight",
        default="sft_vlm",
        type=str,
        help="权重名称前缀（pretrain_vlm, sft_vlm）",
    )
    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="隐藏层维度（512=Small-26M, 768=Base-104M）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=8,
        type=int,
        help="隐藏层数量（Small=8, Base=16）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--temperature",
        default=0.65,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )

    # Common benchmark args
    add_common_args(parser)
    # Override defaults for this script's path convention
    parser.set_defaults(
        data_path="../dataset/LingoQA_Evaluation_data.jsonl",
        image_dir="../dataset/LingoQA_Evaluation_images",
        dump_eval_dir="../benchmark/output",
    )

    args = parser.parse_args()

    # Initialize translator
    translator, backend = init_translator(args.language, backend="googletrans")

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Load test data
    json_data, mapping_data = load_eval_data(args.data_path)

    # Iterate through each QA pair and generate responses
    predictions = []

    for idx, qa_pair in enumerate(json_data):
        prompt = qa_pair["conversations"][0]["content"]
        image_paths = qa_pair["image"]
        pixel_values = load_image(image_paths, args.image_dir, preprocess, args.device)

        if args.use_single_image == 1:
            pixel_values = pixel_values[0:1, :]

        messages = [
            {
                "role": "user",
                "content": prompt.replace("<image>", model.params.image_special_token),
            }
        ]
        inputs_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(
            args.device
        )

        print(f"[图像]: {image_paths}")
        clean_prompt = prompt.replace("\n", "\\n")
        print(f"👶: {clean_prompt}")
        print("🤖️: ", end="")
        # Run generation and decode the resulting token ids to text
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            pixel_values=pixel_values,
        )
        print("\n\n")

        # Decode ONLY the newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Translate response if language is set to English
        if args.language == "english":
            response = translate_response(response, translator, backend)

        predictions.append(collect_prediction(mapping_data, idx, response))

    # Export and evaluate
    export_and_evaluate(predictions, args.dump_eval_dir, args.data_path, args.weight)


if __name__ == "__main__":
    main()
