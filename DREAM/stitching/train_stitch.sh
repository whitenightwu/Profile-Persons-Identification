curr_date=$(date +'%m_%d_%H_%M') 
mkdir -p log
log_file="./log/$curr_date.log"
CUDA_VISIBLE_DEVICES=0,1 python branch_train.py 2>&1 | tee $log_file