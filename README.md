# Teacher-to-Student Knowledge Transfer from Self-Localization Model to Graph-Convolutional Neural Network

This is a PyTorch implementation of the DarkReciprocalRank paper.


## Dependencies
- Pytorch
- DGL
- TensorFlow

## General Setup Instructions
Clone this repo into your current working directory. Then, follow the general setup instructions below:

### Step0: Download Oxford Robotcar Dataset
Download Oxford Robotcar dataset from [here](https://robotcar-dataset.robots.ox.ac.uk/datasets/). And then, converting to color image, and croppig the image with 100 pixels from the left and right and 160 pixels from the bottom removed.

Cropped images should be put in 
```
{working directory}/Reciprocal_rank_KT_GCN/raw_image/{date}/*.png
```

And GPS information must be converted to dictionary type by using 
```
{working directory}/Reciprocal_rank_KT_GCN/multi_view_scene_graph_train/gps_to_dict.py
```

### Step1: Image conversion
#### Canny edge image
```
cd {working directory}/Reciprocal_rank_KT_GCN/image_conversion
python image_edge.py
```
#### Semantic segmentation image
```
cd {working directory}/Reciprocal_rank_KT_GCN/image_conversion/keras-deeplab-v3-plus
python predict_robotcar.py
python semantic_label_to_colorimage.py
```
### Step2: Extracting NetVLAD descriptor
```
cd {working directory}/Reciprocal_rank_KT_GCN/netvlad_tf_open/python
python extract_feature.py
python NetVLAD_feature_to_dict.py
```
### Step3: Generating teacher self-localization model
```
cd {working directory}/Reciprocal_rank_KT_GCN/multi_view_scene_graph_train
python generate_teacher_model.py
```

### Step4: Graph convolution
```
cd {working directory}/Reciprocal_rank_KT_GCN/multi_view_scene_graph_train
graph_conv.py
```
Output is Top-1 accuracy.

## Citation
```
@inproceedings{takeda2021drr,
author = {Koji Takeda and Kanji Tanaka},
title = { Dark Reciprocal-Rank: Teacher-To-Student Knowledge
Transfer from Self-Localization Model to Graph-Convolutional Neural
Network },
booktitle = {{IEEE} International Conference on Robotics and
Automation, {ICRA},
year = {2021},
}
```
