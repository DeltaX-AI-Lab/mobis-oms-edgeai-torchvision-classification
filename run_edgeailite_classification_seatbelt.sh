### ===================================================================================================
# **** Selected models: ****
# https://github.com/TexasInstruments/edgeai-modelzoo/tree/main/models/vision/classification#EdgeAI-TorchVision
### ===================================================================================================
# Model Name              GigaMACs    Top-1 Acc      Description                                Notes
# *mobilenetv2_tv_x1*      0.296	    72.13     not bad, but lite                 ***
# *mobilenetv2_tv_x2_t2*   0.583	    74.57     very good, very lite              ***

# *regnetx400mf_x1*        0.400	    72.70     not bad, but little bit heavy     *** mobilenetv2_tv_x1 better=skip?
# *regnetx800mf_x1*        0.800	    75.20     very good, very lite              ***
# *regnetx1p6gf_x1*        1.600	    77.00     very good, very lite              ***
# *regnetx3p2gf_x1*        3.200	  ? 78.16     very good, very lite              ***

# *resnet50_x1*            4.087	    76.15     very good, but little bit heavy   *** too heavy=skip?
# *resnet50_xp5*                                  should be 4 times lighter then resnet50_x1

# *shufflenetv2_x1p0*             0.151	    69.36       ERROR
# *mobilenetv3_lite_large_x1*     0.213	    72.12       ERROR
# *mobilenetv3_lite_large_x2r*                          ERROR
### ===================================================================================================
### Models summary
### ===================================================================================================
# => Resize = 96, Crop = 96,
# => GFLOPs = 0.106099200, GMACs = 0.053049600, MegaParams = 2.226434,  Top-1-Acc = 72.13   mobilenetv2_tv_x1
# => GFLOPs = 0.164079616, GMACs = 0.082039808, MegaParams = 4.006914,  Top-1-Acc = 74.57   mobilenetv2_tv_x2_t2
# => GFLOPs = 0.142010496, GMACs = 0.071005248, MegaParams = 4.773282,  Top-1-Acc = 72.70   regnetx400mf_x1
# => GFLOPs = 0.289585792, GMACs = 0.144792896, MegaParams = 6.588002,  Top-1-Acc = 75.20   regnetx800mf_x1
# => GFLOPs = 0.588557604, GMACs = 0.294278802, MegaParams = 8.278962,  Top-1-Acc = 77.00   regnetx1p6gf_x1
# => GFLOPs = 1.162649600, GMACs = 0.581324800, MegaParams = 14.28957,  Top-1-Acc = 78.16 ? regnetx3p2gf_x1
# => GFLOPs = 1.458075392, GMACs = 0.729037696, MegaParams = 23.51213,  Top-1-Acc = 76.15   resnet50_x1
# => GFLOPs = 0.364526464, GMACs = 0.182263232, MegaParams = 5.894690,  Top-1-Acc = ??.??   resnet50_xp5
### ===================================================================================================
### # pre-trained weights
### ===================================================================================================
# mobilenetv2_tv_x1     : IMAGENET1K_V1: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
# mobilenetv2_tv_x1     : IMAGENET1K_V2: https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth
# mobilenetv2_tv_x2_t2  : 

# regnetx400mf_x1       : https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth
# regnetx800mf_x1       : https://download.pytorch.org/models/regnet_x_800mf-94a99ebd.pth
# regnetx1p6gf_x1       : https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth
# regnetx3p2gf_x1       : https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth

# resnet50_x1           : https://download.pytorch.org/models/resnet50-19c8e357.pth
# resnet50_x1           : IMAGENET1K_V1: https://download.pytorch.org/models/resnet50-0676ba61.pth
# resnet50_x1           : IMAGENET1K_V2: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
# resnet50_xp5          : ./data/modelzoo/pytorch/image_classification/imagenet1k/jacinto_ai/resnet50-0.5_2018-07-23_12-10-23.pth


#--model_name regnetx400mf_x1_bgr \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth

#--model_name regnetx800mf_x1_bgr \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth

#--model_name regnetx1p6gf_x1_bgr \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth

#--model_name regnetx3p2gf_x1_bgr \
#--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
#--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth


#### ===================================================================================================
##### CUSTOM MAX Scripts    TODO
#### ===================================================================================================

# ************ Finetuning ResNet Models ************
# --rand_scale:     [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -1(None)] 
# --random_erasing: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

### ===================================================================================================
#   SEATBELT
### ===================================================================================================
# ************ resnet50_x1: ************
### The Best Settings (OMS_seatbelt) 93.656 ?
### ===================================================================================================
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name resnet50_x1 \
# --pretrained https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 5 --weight_decay 1e-3 --gpus 3 \
# --exp_name OMS_SB-ep500_ep50-bs512-lr4_lr5-wu5-wd3-0.5_0.5-s
### ===================================================================================================

### =================================================================================================== weights names
# ************ regnetx400mf_x1_bgr: ************
### The Best Settings (OMS_seatbelt) ???%
### ===================================================================================================
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx400mf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 2 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd2-0.5_0.5-s
### ===================================================================================================

### ===================================================================================================
# ************ regnetx800mf_x1_bgr: ************
### The Best Settings (OMS_seatbelt) 92.438 %
### ===================================================================================================
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx800mf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-3 --gpus 0 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd3-0.5_0.5-s
### ===================================================================================================

### ===================================================================================================
# ************ regnetx1p6gf_x1_bgr: ************
### The Best Settings (OMS_seatbelt) 94.223 % !!!
### =================================================================================================== 94.223 %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 2 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd2-0.5_0.5-s
### ===================================================================================================

### ===================================================================================================
# ************ regnetx3p2gf_x1_bgr: ************
### The Best Settings (OMS_seatbelt) 93.600 %
### ===================================================================================================
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx3p2gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-3 --gpus 2 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd3-0.5_0.5-s
### ===================================================================================================




# TESTING

# =============================================================================================================
# =============================================================================================================  v0=91.107 %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.00001 --warmup_epochs 0 --weight_decay 1e-3 --gpus 1 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr5-wu0-wd3-0.5_0.5-s
# =============================================================================================================

# =============================================================================================================  v1=92.438 %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-3 --gpus 1 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd3-0.5_0.5-s
# =============================================================================================================

# =============================================================================================================  v2=94.223 unstable %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.005 --warmup_epochs 0 --weight_decay 1e-3 --gpus 0 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3.5-wu0-wd3-0.5_0.5-s
# =============================================================================================================

# =============================================================================================================  v3=94.166 %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.001 --warmup_epochs 0 --weight_decay 1e-3 --gpus 0 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3-wu0-wd3-0.5_0.5-s
# =============================================================================================================
# =============================================================================================================

# =============================================================================================================  v4=94.477 unstable %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 1 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3-wu0-wd2-0.5_0.5-s
# =============================================================================================================

# =============================================================================================================  v5=94.223 %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 2 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd2-0.5_0.5-s
# =============================================================================================================
# =============================================================================================================

# =============================================================================================================  v6=94.877 super unstable %
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.005 --warmup_epochs 0 --weight_decay 1e-2 --gpus 2 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3.5-wu0-wd2-0.5_0.5-s
# =============================================================================================================











# =============================================================================================================  v4=
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 50 --batch_size 512 --lr 0.001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 0 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep50-bs512-lr0_lr3-wu0-wd2-0.5_0.5-s
# =============================================================================================================


# # =============================================================================================================  v6=
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 50 --batch_size 512 --lr 0.005 --warmup_epochs 0 --weight_decay 1e-2 --gpus 2 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep50-bs512-lr0_lr3.5-wu0-wd2-0.5_0.5-s
# # =============================================================================================================

# # =============================================================================================================  v5=
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 50 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 1 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep50-bs512-lr0_lr4-wu0-wd2-0.5_0.5-s
# # =============================================================================================================


# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --scheduler step --lr 0.001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 0 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3-wu0-wd2-0.5_0.5-s-step_N

# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --scheduler poly --lr 0.001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 1 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3-wu0-wd2-0.5_0.5-s-poly_N


# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr --quantize True \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --scheduler exponential --lr 0.001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 3 \
# --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr3-wu0-wd2-0.5_0.5-s-exponential_N


# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.5 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --lr 0.0001 --warmup_epochs 0 --weight_decay 1e-2 --gpus 0 \
# --exp_name OMS_SB-BGR-ep0_ep500-bs512-lr0_lr4-wu0-wd2-0.5_0.5-s








# SEATBELT
python3 ./references/edgeailite/main/classification/train_classification_main.py \
--dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr \
--pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
--input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
--data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS/ \
--img_resize 96 --img_crop 96 --rand_scale 0.4 1.0 --auto_augument imagenet --random_erasing 0.5 \
--epochs 500 --batch_size 512 --scheduler step --lr 0.001 --warmup_epochs 3 --weight_decay 1e-2 --gpus 0 \
--exp_name OMS_SB-BGR-ep500_ep10-bs512-lr3_lr5-wu3-wd2-0.4_0.5-step


# HOD
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/HOD/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.4 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --scheduler step --lr 0.001 --warmup_epochs 3 --weight_decay 1e-2 --gpus 1 \
# --exp_name OMS_HOD-BGR-ep500_ep10-bs512-lr3_lr5-wu3-wd2-0.4_0.5-step

# SEATBELT_HOD
# python3 ./references/edgeailite/main/classification/train_classification_main.py \
# --dataset_name image_folder_classification --model_name regnetx1p6gf_x1_bgr \
# --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth \
# --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125 \
# --data_path /hdd/max/TI/seatbelt/final_datasets/SEATBELTS_HOD/ \
# --img_resize 96 --img_crop 96 --rand_scale 0.4 1.0 --auto_augument imagenet --random_erasing 0.5 \
# --epochs 500 --batch_size 512 --scheduler step --lr 0.001 --warmup_epochs 3 --weight_decay 1e-2 --gpus 3 \
# --exp_name OMS_SBHOD-BGR-ep500_ep10-bs512-lr3_lr5-wu3-wd2-0.4_0.5-step







# 'cosine'				# 'poly', 'step', 'exponential', 'cosine'

# NOTES:
# resnet50 can keep performance well QAT
# regnetx1p6gf_x1_bgr   reqire more tests
# regnetx3p2gf_x1_bgr
# recommend RegNetX


# Weight decay is applied to all layers / parameters and that weight decay factor is good.
# Ensure that the Convolution layers in the network have Batch Normalization layers immediately after that.
# The only exception allowed to this rule is for the very last Convolution layer in the network
# (for example the prediction layer in a segmentation network or detection network,
# where adding Batch normalization might hurt the floating point accuracy).

# v1,v2
# Fixed point mode, especially the 8-bit mode can have accuracy degradation.
# The tools and guidelines provided here help to avoid accuracy degradation with quantization.

# If you are getting accuracy degradation with 8-bit inference, the first thing to check is 16-bit inference.
# If 16-bit inference provides accuracy close to floating point and 8-bit has an accuracy degradation,
# there it is likely that the degradation si due to quantization.
# However, if there is substantial accuracy degradation with 16-bit inference itself,
# then it is likely that there is some issue other than quantization.


# Test:
# args.weight_decay
# args.bias_decay = args.weight_decay
# args.lr_clips = None --> 0.00001

#  --exp_name OMS_SB-BGR-QAT-ep0_ep500-bs512-lr0_lr4-wu0-wd3-0.5_0.5-s

# 0.0001 < lr < 0.001   lr = 0.005
# 0.001 < wd < 0.01     wd = 0.01
