MA-YOLO

Official PyTorch implementation of MA-YOLO, a modified YOLO-based object detector for traffic participant detection in autonomous driving scenarios.
This repository is released to support the reproducibility of the experimental results reported in our journal submission.

1. Introduction

MA-YOLO is built upon YOLOv5 and introduces several improvements to enhance detection performance for small and occluded traffic objects (e.g., pedestrians and cyclists), while maintaining real-time efficiency.

Main contributions include:
- A modified bounding box regression loss (MPMIoU)
- An atrous decoupled detection head
- A small-object detection branch
- A multi-scale feature extraction module (MSC3)

2. Supported Datasets

MA-YOLO is evaluated on public autonomous driving datasets:

KITTI: http://www.cvlibs.net/datasets/kitti/
nuScenes: https://www.nuscenes.org/

Note: 
Due to license and size constraints, datasets and trained weights are NOT included in this repository.

3. Repository Structure

MA-YOLO/
├── train.py                 # training script
├── test.py                  # evaluation script
├── detect.py                # inference script
├── models/                  # model definitions and YAML configs
│   ├── common.py
│   ├── yolo.py
│   ├── experimental.py
│   └── *.yaml
├── utils/                   # dataset loading, loss, metrics, utilities
│   ├── datasets.py
│   ├── loss.py
│   ├── metrics.py
│   └── torch_utils.py
├── datasets/                # user-prepared datasets (not included)
├── weights/                 # trained weights (not included)
└── README.md

4. Environment Setup

- Python >= 3.8
- PyTorch >= 1.10
- CUDA 11.x (recommended)

5. Dataset Preparation
Download the datasets from the official websites and organize them in YOLO format:

datasets/
├── kitti/
│   ├── images/
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── labels/
│       ├── train
│       ├── val
│       └── test
└── nuscenes_01/
    ├── images/
    │   ├── train
    │   ├── val
    │   └── test
    └── labels/
        ├── train
        ├── val
        └── test

6. Training
Example command for training on KITTI:

python train.py \
  --data data/kitti.yaml \
  --cfg models/yolov5s_head_atrousdecouple_BBMSC3.yaml \
  --batch-size 4 \
  --epochs 300 \
  --img 640
Training logs and checkpoints will be saved in runs/train/.

7. Evaluation
Evaluate a trained model:

python test.py \
  --data data/kitti.yaml \
  --weights weights/best.pt \
  --img 640
Metrics include Precision, Recall, mAP@0.5, and mAP@0.5:0.95.

8. Notes
Experiments were conducted on an NVIDIA RTX 3060 Ti GPU.
Dataset splits and evaluation protocols follow the paper.
Results reported in the paper can be reproduced using this code and public datasets.

9. Contact
For questions regarding reproduction or implementation details, please contact the corresponding author.
