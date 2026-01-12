import json
import os
import cv2
import pandas as pd
import sys

# Ensure repo root is on sys.path so `benchmark` package can be imported when
# running this script from the `scripts/` directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from googletrans import Translator
from utils import safe_translate
from benchmark.constants import Keys


# =========================
# Translation cache
# =========================
translator = Translator()
translation_cache = {}


def cached_translate(text):
    if text not in translation_cache:
        translation_cache[text] = safe_translate(
            translator, text, src="en", dest="zh-cn"
        )
    return translation_cache[text]


# =========================
# Config
# =========================
# subset = "Action"
# mode = "train"
# subset = "Scenery"
# mode = "train"
subset = "Evaluation"
mode = "val"

base_path = f"/mnt/sdb/Download_data/LingoQA/{subset}"
image_base_path = os.path.join(base_path, "images", mode)
parquet_file_path = os.path.join(base_path, f"{mode}.parquet")

save_base_path = "/mnt/sdb/PROJECTS/minimind-v/dataset"
save_image_path = os.path.join(save_base_path, f"LingoQA_{subset}_images")
save_jsonl_path = os.path.join(save_base_path, f"LingoQA_{subset}_data.jsonl")
mapping_jsonl_path = save_jsonl_path.replace(".jsonl", "_mapping.jsonl")

os.makedirs(save_image_path, exist_ok=True)


# =========================
# Load parquet & group
# =========================
print("Loading parquet...")
df = pd.read_parquet(parquet_file_path)

# Use enum .value (string) for DataFrame column operations to avoid Enum
# objects becoming column labels (which show up as `Keys.xxx`).
df = df[
    [
        Keys.question_id.value,
        Keys.segment_id.value,
        Keys.question.value,
        Keys.answer.value,
    ]
]

df = (
    df.groupby([Keys.question_id.value, Keys.segment_id.value, Keys.question.value])
    .agg(list)
    .reset_index()
)

# Group once (BIG speedup)
df_groups = dict(tuple(df.groupby("segment_id")))

seq_names = sorted(os.listdir(image_base_path))


# =========================
# Main loop
# =========================
with open(save_jsonl_path, "w", encoding="utf-8") as fout, open(
    mapping_jsonl_path, "w", encoding="utf-8"
) as fmap:
    for seq_name in tqdm(seq_names, desc="Processing sequences", unit="sequence"):
        seq_path = os.path.join(image_base_path, seq_name)
        if not os.path.isdir(seq_path):
            continue

        seq_df = df_groups.get(seq_name)
        if seq_df is None or seq_df.empty:
            continue

        # -------- Images --------
        image_files = sorted(
            f
            for f in os.listdir(seq_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        image_names = []
        for image_file in image_files:
            image_file_path = os.path.join(seq_path, image_file)
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

            if image is None:
                continue

            h, w = image.shape[:2]
            resized_image = cv2.resize(image, (w // 4, h // 4))

            frame_idx = int(os.path.splitext(image_file)[0].split("_")[-1])
            image_name = f"{seq_name}_{frame_idx:05d}.jpg"
            output_path = os.path.join(save_image_path, image_name)

            cv2.imwrite(output_path, resized_image)
            image_names.append(image_name)

        if not image_names:
            continue

        all_new_image_paths = ",".join(image_names)

        # -------- QA → JSONL --------
        for _, row in seq_df.iterrows():
            question_eng = row["question"]
            answer_eng = row["answer"]
            question_id = row["question_id"]

            question_ch = cached_translate(question_eng)
            if isinstance(answer_eng, str):
                answer_ch = cached_translate(answer_eng)
            elif isinstance(answer_eng, list):
                # for evaluation set where multiple answers exist
                answer_ch = [cached_translate(a) for a in answer_eng]

            jsonl_entry = {
                "conversations": [
                    {"role": "user", "content": f"{question_ch}\n<image>"},
                    {"role": "assistant", "content": answer_ch},
                ],
                "image": all_new_image_paths,
            }

            fout.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

            # Mapping entry (stable & explicit)
            fmap.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "segment_id": seq_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


print(f"\n✅ Done")
print(f"Data saved to: {save_jsonl_path}")
print(f"Mapping saved to: {mapping_jsonl_path}")
