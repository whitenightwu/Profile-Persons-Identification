#!/usr/bin/env bash
curr_date=$(date +'%m_%d_%H_%M')
mkdir -p log
log_file="./log/$curr_date.log"

CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --root_path="/home/u0060/Datasets/cropped/" \
    --resume="/home/u0060/Profile-Persons-Identification/LightCNN/without/lightCNN_130_checkpoint.pth.tar"
    --img_list="val_list.txt" \
    --model="LightCNN-9" \
    --num_classes=4382 2>&1 | tee ${log_file}

