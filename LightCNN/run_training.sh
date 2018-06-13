#!/usr/bin/env bash

curr_date=$(date +'%m_%d_%H_%M') 
log_file="./$curr_date.log"

# train the model with GPUs 0, 1, 2, and 3
CUDA_VISIBLE_DEVICES=0,1 python main.py \
	--root_path=/home/u0060/Datasets/msceleb_subset/image \
	--train_list=/home/u0060/Datasets/msceleb_subset/train_list.txt \
	--val_list=/home/u0060/Datasets/msceleb_subset/test_list.txt \
	--model="LightCNN-29v2" --num_classes=10 