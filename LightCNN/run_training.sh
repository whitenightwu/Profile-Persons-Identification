#!/usr/bin/env bash
curr_date=$(date +'%m_%d_%H_%M')
mkdir -p log
log_file="./log/$curr_date.log"

CUDA_VISIBLE_DEVICES=0,1 python main.py --root_path="/home/u0060/Datasets/cropped/" \
	--train_list="train.txt" \
	--val_list="val.txt" \
	--end2end \
	--model="LightCNN-9" --num_classes=4382 2>&1 | tee ${log_file}

