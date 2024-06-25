#!/usr/bin/env bash
python3 main.py --resume /netscratch/naeem/detr-r50-coco.pth --batch_size 32 --num_workers 20 --dataset_file phenobench --output_dir /netscratch/naeem/phenobench_detr_r50_real_baseline --epochs 100 
