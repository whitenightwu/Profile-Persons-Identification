#!/usr/bin/env bash
curr_date=$(date +'%m_%d_%H_%M')
mkdir -p log
log_file="./log/$curr_date.log"

CUDA_VISIBLE_DEVICES=0,1 python main.py --root_path="/home/u0060/Datasets/cropped/" \
	--train_list="train.txt" \
	--val_list="val.txt" \
	--model="LightCNN-9" --num_classes=4382 --end2end 2>&1 | tee ${log_file}

