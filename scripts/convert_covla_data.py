import os
import json
import cv2

from typing import List, Dict, Any
from utils import normalize, jaccard_similarity
from tqdm import tqdm

from googletrans import Translator
from utils import safe_translate


def load_jsonl_safely(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file that may contain concatenated JSON objects
    (e.g. {...}{...} without newlines).

    Args:
        path: Path to .jsonl file

    Returns:
        List of parsed JSON objects (dicts)
    """
    decoder = json.JSONDecoder()
    objects = []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    idx = 0
    length = len(text)

    while idx < length:
        # Skip whitespace and newlines
        while idx < length and text[idx].isspace():
            idx += 1

        if idx >= length:
            break

        try:
            obj, next_idx = decoder.raw_decode(text, idx)
            objects.append(obj)
            idx = next_idx
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed at char {idx}") from e

    return objects


def downsample_captions_non_ml(
    rich_captions: dict,
    max_samples: int = 30,
    min_samples: int = 1,
    similarity_threshold: float = 0.7,
):
    """
    Downsample captions using token-based Jaccard similarity.

    Args:
        rich_captions (dict): {frame_id: caption}
        similarity_threshold (float): higher = stricter filtering

    Returns:
        dict: downsampled captions
    """

    if not rich_captions:
        return {}

    keys = list(rich_captions.keys())
    captions = list(rich_captions.values())

    tokenized = [normalize(c) for c in captions]

    selected = [0]  # always keep first frame

    for i in range(1, len(captions)):
        if len(selected) >= max_samples:
            break

        is_similar = False
        for j in selected:
            if jaccard_similarity(tokenized[i], tokenized[j]) >= similarity_threshold:
                is_similar = True
                break

        if not is_similar:
            selected.append(i)

    # Guarantee minimum samples
    if len(selected) < min_samples:
        selected = list(range(min(min_samples, len(captions))))

    return {keys[i]: captions[i] for i in selected}


base_path = "/mnt/sdb/Download_data/CoVLA-Dataset"
video_path = os.path.join(base_path, "videos")
caption_path = os.path.join(base_path, "captions")
save_base_path = "/mnt/sdb/PROJECTS/minimind-v/dataset"
save_image_path = os.path.join(save_base_path, "covla_images")
save_caption_file = os.path.join(save_base_path, "covla_data.jsonl")
video_fps = 20  # fps value for video frames

total_saved_frames = 0

translator = Translator()

video_names = sorted(os.listdir(video_path))
# Ensure save directories exist
os.makedirs(save_image_path, exist_ok=True)
os.makedirs(save_base_path, exist_ok=True)


for video_name in tqdm(video_names, desc="Processing videos", unit="video"):
    # Load the video
    video_file_path = os.path.join(video_path, video_name)
    video = cv2.VideoCapture(video_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    seq_name = os.path.splitext(video_name)[0]
    # Load corresponding captions for this video folder
    caption_file_path = os.path.join(caption_path, f"{seq_name}.jsonl")
    # read captions from jsonl file
    caption_data = load_jsonl_safely(caption_file_path)

    # Extract the rich captions from the loaded data
    rich_captions = {
        k: v["rich_caption"] for entry in caption_data for k, v in entry.items()
    }

    # From the rich_captions, we will down sample frames based on the similarity of captions
    downsampled_captions = downsample_captions_non_ml(
        rich_captions,
        max_samples=30,
        min_samples=1,
        similarity_threshold=0.7,
    )

    # print(
    #     f"Processing video: {video_name}, total frames: {total_frames}, selected captions: {len(downsampled_captions)}"
    # )
    total_saved_frames += len(downsampled_captions)

    # Based on the indices of downsampled captions, extract and save the corresponding frames
    caption_outputs = []

    for frame_id in downsampled_captions.keys():
        frame_idx = int(frame_id)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # Save images
        ret, frame = video.read()

        # Reduce the resolution of image by fourths
        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))

        if ret:
            image_name = f"{seq_name}_{frame_idx:05d}.jpg"
            output_path = os.path.join(save_image_path, image_name)
            cv2.imwrite(output_path, frame)

            # Create jsonl entry
            # Example entry: {"conversations": [{"role": "user", "content": "提供给定图像的简要描述。\n<image>"}, {"role": "assistant", "content": "橄榄油是自由使用的健康成分。"}], "image": "GCC_train_002582585.jpg"}

            # translate the caption to chinees
            caption_english = downsampled_captions[
                frame_id
            ]  # Placeholder for translation function
            caption_chinese = safe_translate(
                translator, caption_english, src="en", dest="zh-cn"
            )
            caption_entry = {
                "conversations": [
                    {"role": "user", "content": "提供给定图像的描述。\n<image>"},
                    {"role": "assistant", "content": caption_chinese},
                ],
                "image": image_name,
            }
            caption_outputs.append(caption_entry)

    # Save caption outputs to jsonl file
    # for first video, create/overwrite file; for subsequent videos, append
    mode = "w" if video_name == video_names[0] else "a"
    with open(save_caption_file, mode, encoding="utf-8") as f:
        for entry in caption_outputs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # close video
    video.release()

print(f"Total saved frames with captions: {total_saved_frames}")
