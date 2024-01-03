#!/usr/bin/env bash

# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

# Date/time in YYYY-MM-DD_HH-mm-SS format
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`

#=========================================================================================
# *** Supported models ***
#=========================================================================================
# model=regnet_x_1_6gf

#=========================================================================================
# *** Supported pre-trained weights ***
#=========================================================================================
# model_weights="RegNet_X_1_6GF_Weights.IMAGENET1K_V2"
#=========================================================================================

model=regnet_x_1_6gf							
model_weights="RegNet_X_1_6GF_Weights.IMAGENET1K_V2"
data_path="/hdd/max/TI/seatbelt/final_dataset/final_dataset_videos/"

epochs=500									# [20, 50, 100, 300, 500]
batch_size=352								# [32, 64, 128, 256, 512]
weight_decay=0.001							# [0.001, 0.0001, 0.00001]
lr=0.0001									# [0.001, 0.0001, 0.00001]
lr_warmup_epochs=0							# [0, 5, 10]
label_smoothing=0.1							# [0.0, 0.1, 0.11]
lr_scheduler=cosineannealinglr				# [steplr, cosineannealinglr, exponentiallr]
opt=sgd										# [adamw, rmsprop, sgd]

train_crop_size=96							# [96, 128, 160, 192, 224, 256]
val_resize_size=96							# [96, 128, 160, 192, 224, 256] + extra
val_crop_size=96							# [96, 128, 160, 192, 224, 256]

model_surgery=2								# [0, 1, 2]
quantization=2								# [0, 1, 2]
quantization_type=WT8SP2_AT8SP2				# [WT8SP2_AT8SP2, WC8_AT8]

auto_augment=imagenet						# [imagenet, cifar10, svhn]
random_erase=0.5							# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
mixup_alpha=0.0								# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
cutmix_alpha=0.0							# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

opset_version=11							# [9, 11]

# exp_name="v1"
exp_name="${epochs}-${batch_size}-${weight_decay}-${lr}-${lr_warmup_epochs}-${train_crop_size}-${val_resize_size}-${val_crop_size}-${model_surgery}-${quantization}-${quantization_type}-${opset_version}"
output_dir="./data/checkpoints/torchvision/${DATE_TIME}_imagenet_classification_${model}_${exp_name}"

#====================================================================================================================
command="./references/classification/train.py \
--model=$model --weights=$model_weights --data-path=$data_path --output-dir=$output_dir \
--model-surgery=$model_surgery --quantization=$quantization --quantization-type=$quantization_type \
--epochs=$epochs --batch-size=$batch_size --lr=$lr --lr-warmup-epochs=$lr_warmup_epochs \
--weight-decay=$weight_decay --label-smoothing=$label_smoothing --lr-scheduler=$lr_scheduler --opt=$opt \
--train-crop-size=$train_crop_size --val-resize-size=$val_resize_size --val-crop-size=$val_crop_size \
--auto-augment=$auto_augment --random-erase=$random_erase --mixup-alpha=$mixup_alpha --cutmix-alpha=$cutmix_alpha \
--opset-version=$opset_version"
#====================================================================================================================

# [ --model-ema --cache-dataset --use-deterministic-algorithms ]
python3 ${command} --device=cuda:0 --cache-dataset --use-deterministic-algorithms
