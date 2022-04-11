# Adversarial Robustness of Deep Sensor Fusion Models

This is a repo for our WACV2022 submission [Adversarial Robustness of Deep Sensor Fusion Models](arxiv.org/abs/2006.13192).

The code is based on [mmdetection package](https://github.com/open-mmlab/mmdetection) and [Universenet repo](https://github.com/shinya7y/UniverseNet), with modification in `mmdetection/mmdet/datasets/pipelines/` for LiDAR channel data loading; in `mmdetection/mmdet/models/` for YOLOv4 implementation; in `mmdetection/mmdet/apis/` for adversarial attack api. Please follow the original instructions to install the package and run the training/testing process. 
Waymo dataset can be downloaded and installed following their [instructions](https://github.com/waymo-research/waymo-open-dataset).

Configuration files for yolo models are stored in `configs/waymo_open/mosaic`. We can use `--attack` argument in `tools/test.py` for adversarial attack, and `--adv_train` option in `tools/train.py` for adversarial training.

