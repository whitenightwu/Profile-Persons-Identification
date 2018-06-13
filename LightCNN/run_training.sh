#!/usr/bin/env bash
curr_date=$(date +'%m_%d_%H_%M')
mkdir -p log
log_file="./log/$curr_date.log"

CUDA_VISIBLE_DEVICES=0,1 python main.py --root_path="/home/u0060/Datasets/msceleb_subset/image" \
	--train_list="/home/u0060/Datasets/msceleb_subset/train_list.txt" \
	--val_list="/home/u0060/Datasets/msceleb_subset/test_list.txt" \
	--model="LightCNN-29v2" --num_classes=10 2>&1 | tee ${log_file}