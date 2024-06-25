#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --resume /netscratch/naeem/detr-r50-coco.pth --batch_size 32 --num_workers 10 --dataset_file phenobench --output_dir /netscratch/naeem/phenobench_detr_r50_syn_v6 --epochs 100 
