"""Shared utilities for benchmark evaluation scripts."""

import os
import json
import time
import argparse

import torch
import pandas as pd

from benchmark.evaluate import evaluate


def load_jsonl_data(jsonl_path):
    """Load data from a JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def load_eval_data(data_path):
    """Load evaluation data and its mapping file, with length validation."""
    json_data = load_jsonl_data(data_path)
    mapping_data = load_jsonl_data(data_path.replace(".jsonl", "_mapping.jsonl"))
    assert len(json_data) == len(mapping_data), (
        f"Data and mapping length mismatch! "
        f"Data: {len(json_data)}, Mapping: {len(mapping_data)}"
    )
    return json_data, mapping_data


def add_common_args(parser):
    """Add benchmark arguments shared by all evaluation scripts."""
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/LingoQA_Evaluation_data.jsonl",
        help="Test data path (JSONL)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./dataset/LingoQA_Evaluation_images",
        help="Test image directory",
    )
    parser.add_argument(
        "--dump_eval_dir",
        type=str,
        default="./benchmark/output",
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device to run on",
    )
    parser.add_argument(
        "--use_single_image",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to use single image mode (0=no, 1=yes)",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--language",
        default="english",
        type=str,
        choices=["english", "chinese"],
        help="Output language (english=English, chinese=Chinese), default English to match LingoQA benchmark",
    )


def init_translator(language, backend="deep_translator"):
    """Initialize a translator for Chinese-to-English conversion.

    Args:
        language: Target output language ("english" or "chinese").
        backend: Translation library ("googletrans" or "deep_translator").

    Returns:
        A (translator, backend) tuple, or (None, backend) if not needed.
    """
    if language != "english":
        return None, backend

    if backend == "googletrans":
        try:
            from googletrans import Translator

            translator = Translator()
            print(
                "Translator initialized (googletrans) for Chinese -> English conversion"
            )
            return translator, backend
        except ImportError:
            print(
                "Warning: googletrans not installed. Run: pip install googletrans==4.0.0-rc1"
            )
            return None, backend
    elif backend == "deep_translator":
        try:
            from deep_translator import GoogleTranslator

            translator = GoogleTranslator(source="zh-CN", target="en")
            print(
                "Translator initialized (deep_translator) for Chinese -> English conversion"
            )
            return translator, backend
        except ImportError:
            print(
                "Warning: deep-translator not installed. Run: pip install deep-translator"
            )
            return None, backend
    else:
        raise ValueError(f"Unknown translation backend: {backend}")


def translate_response(response, translator, backend="deep_translator"):
    """Translate a response string. Returns original text on failure."""
    if translator is None:
        return response

    if backend == "googletrans":
        from scripts.utils import safe_translate

        translated = safe_translate(translator, response, src="zh-cn", dest="en")
        print(f"Translated to English: {translated}")
        return translated
    elif backend == "deep_translator":
        try:
            translated = translator.translate(response)
            print(f"Translated to English: {translated}")
            return translated
        except Exception as e:
            print(f"Translation failed: {e}, keeping original response")
            return response
    return response


def collect_prediction(mapping_data, idx, response):
    """Build a prediction record from mapping data and a response."""
    return {
        "question_id": mapping_data[idx]["question_id"],
        "segment_id": mapping_data[idx]["segment_id"],
        "answer": response,
    }


def export_and_evaluate(predictions, dump_eval_dir, data_path, suffix):
    """Export predictions to CSV and run LingoQA evaluation."""
    os.makedirs(dump_eval_dir, exist_ok=True)

    predictions_csv_path = os.path.join(
        dump_eval_dir,
        os.path.basename(data_path).replace(".jsonl", f"_predictions_{suffix}.csv"),
    )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nPredictions saved to {predictions_csv_path}")

    print("\nEvaluating predictions...")
    evaluate.callback(predictions_csv_path, 16, True)


class ProgressTracker:
    """Track and display progress with ETA for benchmark evaluation loops."""

    def __init__(self, total):
        self.total = total
        self.start_time = time.time()

    def update(self, idx, image_paths, prompt):
        """Print progress info for the current sample."""
        elapsed = time.time() - self.start_time
        avg_time = elapsed / (idx + 1) if idx > 0 else 0
        eta = avg_time * (self.total - idx - 1)

        print(
            f"\n[{idx+1}/{self.total}] ETA: {eta/60:.1f}min | "
            f"Avg: {avg_time:.1f}s/sample"
        )
        print(f"Images: {image_paths}")
        clean_prompt = prompt.replace("\n", "\\n")
        print(f"Q: {clean_prompt[:100]}...")
        print("A: ", end="")
