Run pretrain on v1 data

Run sft on v1 data
```
python train_sft_vlm.py --save_weight sft_vlm_v1 --max_num_images 5 --use_bucketed_sampler 1 --data_path ../dataset/sft_data_v1.jsonl --images_path ../dataset/sft_images_v1 --from_weight pretrain_vlm_v1 --use_wandb
```

Evaluate model

Benchmark with Qwen3-VL 8B Instruct model
```
conda activate qwen3

python run_qwen3_benchmark.py
```

SFT model with mutiple images as input
```
python run_benchmark_eval.py --weight sft_vlm_v1
```

Pretrin model with single image as input
```
python ./run_benchmark_eval.py --weight pretrain_vlm_v1 --use_single_image 1
```

Lora finetune Qwen3
```
python qwen3/train_qwen3_lora.py \
--lora_r 4 \
--lora_alpha 8 \
--batch_size 2 \
--gradient_accumulation_steps 4 \
--max_samples 10000
```

Evaluate Qwen3 lora finetuned model
```
python benchmark/run_qwen3_lora_benchmark.py
```