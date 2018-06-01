# prepare CFP data first
dst_dir=../../data/CFP
list_file_name=../../../Datasets/cfp-dataset/Data/list_name.txt
dst_file_name=align_img_list.txt
python pre_cfp_data.py $dst_dir $list_file_name $dst_file_name


# align CFP image
pnp_file=pnp.txt
image_prefix=../../../Datasets/cfp-dataset/
alignment_file=cfp_alignment.txt
aligned_img_file=align_img_list.txt
pose_file=estimate_pose.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/u0060/anaconda3/envs/py36/lib
./test_process_align $pnp_file $image_prefix $alignment_file $aligned_img_file $pose_file
