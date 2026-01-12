Run pretrain on v1 data

Run sft on v1 data
```
python train_sft_vlm.py --save_weight sft_vlm_v1 --max_num_images 5 --use_bucketed_sampler 1 --data_path ../dataset/sft_data_v1.jsonl --images_path ../dataset/sft_images_v1 --from_weight pretrain_vlm_v1 --use_wandb
```

Evaluate model
```
python run_benchmark_eval.py --weight sft_vlm_v1
```