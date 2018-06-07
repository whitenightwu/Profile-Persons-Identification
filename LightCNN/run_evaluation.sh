python extract_features.py --resume="LightCNN_29Layers_V2_checkpoint.pth.tar" \
--root_path="/home/u0060/Datasets/msceleb_subset/image" \
--img_list="img_list.txt" \
--model="LightCNN-29v2" \
--num_classes=80013 \
--save_path="features"