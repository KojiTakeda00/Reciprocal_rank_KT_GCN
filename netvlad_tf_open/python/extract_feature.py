# This takes a lot of time to run, so not written as unit test.

import netvlad_tf.nets as nets

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.spatial.distance as scid
import scipy.signal as scisig
import tensorflow as tf
import time
import unittest

import sys
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from netvlad_tf.image_descriptor_jpg import ImageDescriptor as ImageDescriptor_jpg
from netvlad_tf.image_descriptor_png import ImageDescriptor as ImageDescriptor_png
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.precision_recall as pr

from tqdm import tqdm

tf.reset_default_graph()


date1 = "2015-08-28-09-50-22"
date2 = "2015-10-30-13-52-14"

target_dates = [date1, date2]
for date in target_dates:
    for ic in ["Raw", "canny", "semantic"]:
        if ic == "Raw":
            path_to_image_dir = "../../raw_image/" + date + "/"
            imd = ImageDescriptor_png(is_grayscale=False)
        if ic == "canny":
            path_to_image_dir = "../../image_conversion/canny_edge_img/" + date + "/"
            imd = ImageDescriptor_png(is_grayscale=False)
        if ic == "semantic":
            path_to_image_dir = "../../image_conversion/keras-deeplab-v3-plus/result/" + date + "/segmentation_result_color/"
            imd = ImageDescriptor_jpg(is_grayscale=False)
        feats = imd.describeAllJpegsInPath(path_to_image_dir, 4, verbose=False)
        #%%
        pbar = tqdm(total=len(glob.glob(path_to_image_dir + "/*")))
    
        os.makedirs("netvlad_feature_" + ic + "/"+date,exist_ok = True)
        
        for feat,img_path in zip(feats,sorted(glob.glob(path_to_image_dir +  "/*"))):
            pbar.update(1)
            png_name = os.path.basename(img_path)
            name = png_name.split(".")[0]
            np.savetxt("netvlad_feature_" + ic + "/" +date + "/" + name +".txt",feat)
        
        feats = []
        pbar.close()





















