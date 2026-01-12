import os
import sys

__package__ = "evaluator"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import warnings
import torch
import json
import pandas as pd

from PIL import Image
from trainer.trainer_utils import setup_seed
from model.model_vlm import MiniMindVLM, VLMConfig
from benchmark.evaluate import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

warnings.filterwarnings("ignore")


def load_jsonl_data(jsonl_path):
    """Load data from a JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line.strip())
            data.append(item)
    return data


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
        "--use_single_image",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用单图像模式（0=否，1=是）",
    )
    parser.add_argument("--max_new_tokens", default=512, type=int, help="最大生成长度")
    parser.add_argument(
        "--temperature",
        default=0.65,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/LingoQA_Evaluation_data.jsonl",
        help="测试数据路径",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="../dataset/LingoQA_Evaluation_images",
        help="测试图像目录",
    )
    parser.add_argument(
        "--dump_eval_dir",
        type=str,
        default="../benchmark/output",
        help="Eval output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 自动测试data_path中的所有test case

    # Load test jsonl data
    json_data = load_jsonl_data(args.data_path)
    # Load test jsonl mapping data
    mapping_data = load_jsonl_data(args.data_path.replace(".jsonl", "_mapping.jsonl"))

    assert (
        len(json_data) == len(mapping_data)
    ), f"数据与映射长度不匹配！, 数据长度: {len(json_data)}, 映射长度: {len(mapping_data)}"

    # Iterate through each QA pair and generate responses
    # Record the predictions with the columns question_id, segment_id and answer
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
        # Run generation and decode the resulting token ids to text (Chinese)
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

        question_id = mapping_data[idx]["question_id"]
        segment_id = mapping_data[idx]["segment_id"]

        predictions.append(
            {
                "question_id": question_id,
                "segment_id": segment_id,
                "answer": response,
            }
        )

    # Export predictions to a predictions.csv file
    if os.path.exists(args.dump_eval_dir) is False:
        os.makedirs(args.dump_eval_dir)

    predictions_csv_path = os.path.join(
        args.dump_eval_dir,
        os.path.basename(args.data_path).replace(
            ".jsonl", f"_predictions_{args.weight}.csv"
        ),
    )

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")
    print(f"Predictions saved to {predictions_csv_path}")

    # Evaluate the predictions (call the click-wrapped function's callback directly)
    evaluate.callback(predictions_csv_path, 16)


if __name__ == "__main__":
    main()
