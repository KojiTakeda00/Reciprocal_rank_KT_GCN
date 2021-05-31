# Teacher-to-Student Knowledge Transfer from Self-Localization Model to Graph-Convolutional Neural Network

This is a PyTorch implementation of the DarkReciprocalRank paper.

Preprint can be found [here](https://arxiv.org/pdf/2011.00402.pdf).

![image](https://user-images.githubusercontent.com/52208935/120152979-211e1780-c229-11eb-8441-374e5c01a9f0.png)

This codebase supports the use of dark reciprocal rank for GCN-based visual place classification (VPC). In other words, the self-localization system is used as a teacher to transfer knowledge to the student GCN. The teacher self-localization system is generally modeled as a ranking function. Therefore, this approach could be generalized to the various types of existing teacher self-localization systems. 

This implementation focuses on the multi-view-based VPC scenario covered in the ICRA2021 paper. This scenario showed the highest performance in the experiment. The system that reproduces all the ablation studies handled in the experiment has not been implemented yet and is a future task. In the current version systems, it first downloads and formats the dataset from the Oxford RobotCar dataset. Next, it does some third-party image filtering to create a multi-channel, multi-view scene graph. Next, it performs NetVLAD feature extraction to implement the teacher system. Next, it uses DGL to knowledge transfer and training/testing of GCN.

**Results on Oxford RobotCar dataset**

![image](https://user-images.githubusercontent.com/52208935/120153724-08fac800-c22a-11eb-9e5d-1b750ad71dee.png)
![image](https://user-images.githubusercontent.com/52208935/120153643-eec0ea00-c229-11eb-9d69-217a95080ec5.png)
![image](https://user-images.githubusercontent.com/52208935/120153683-fb454280-c229-11eb-8f5f-bf4364305f5a.png)


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
python graph_conv.py
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
