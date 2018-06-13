#!/usr/bin/env bash
curr_date=$(date +'%m_%d_%H_%M')
mkdir -p log
log_file="./log/$curr_date.log"

CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --resume="" \
    --root_path="/home/u0060/Datasets/msceleb_subset/image" \
    --img_list="img_list.txt" \
    --model="LightCNN-9" \
    --num_classes=4382 2>&1 | tee ${log_file}