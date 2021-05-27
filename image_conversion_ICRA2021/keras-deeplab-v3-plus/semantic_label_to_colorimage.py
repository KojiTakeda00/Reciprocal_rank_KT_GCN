import numpy as np
import cv2
from tqdm import tqdm
import glob
import os
import sys


date = "2015-11-10-10-32-52"




semantic_label_to_ID ={"load":0,
                       "sidewalk":1,
                       "building":2,
                       "wall":3,
                       "fence":4,
                       "pole":5,
                       "traffic_light":6,
                       "traffic_sign":7,
                       "vegetation":8,
                       "terrain":9,
                       "sky":10,
                       "person":11,
                       "rider":12,
                       "car":13,
                       "truck":14,
                       "bus":15,
                       "train":16,
                       "motorcycle":17,
                       "bicycle":18}




ID_to_semantic_label = {0:"load",
                        1:"sidewalk",
                        2:"building",
                        3:"wall",
                        4:"fence",
                        5:"pole",
                        6:"traffic_light",
                        7:"traffic_sign",
                        8:"vegetation",
                        9:"terrain",
                        10:"sky",
                        11:"person",
                        12:"rider",
                        13:"car",
                        14:"truck",
                        15:"bus",
                        16:"train",
                        17:"motorcycle",
                        18:"bicycle"}




semantic_label_to_BGR = {"load":(128, 64,128),
                       "sidewalk":(232, 35,244),
                       "building":( 70, 70, 70),
                       "wall":(156,102,102),
                       "fence":(153,153,190),
                       "pole":(153,153,153),
                       "traffic_light":(30,170, 250),
                       "traffic_sign":(0,220,220),
                       "vegetation":(35,142, 107),
                       "terrain":(152,251,152),
                       "sky":( 180,130,70),
                        "person":(60, 20, 220),
                       "rider":(0,  0,  255),
                       "car":(  142,  0,0),
                       "truck":(  70,  0, 0),
                       "bus":(  100, 60,0),
                       "train":(  100, 80,0),
                       "motorcycle":(  230,  0,0),
                         "bicycle":(32, 11, 119)}


all_label_in_cityscapes = 19
"""
def generate_semantic_color_image_from_npz(npz_path,save_path):
    #convert segmentation result to color image
    #[512,512,3],each element indicate probability of semantic class 0 to 100
    npz = np.load(npz_path)
    color = np.zeros((512,512,3))
    semantic_result = np.argmax(npz["arr_0"],-1)

    for semantic_ID in range(all_label_in_cityscapes):
        color_map = semantic_label_to_BGR[ID_to_semantic_label[semantic_ID]]
        color[np.where((semantic_result == semantic_ID))] = color_map
        
    color_resized = cv2.resize(color,(1080,800))
    cv2.imwrite(save_path,color_resized)
"""

def generate_semantic_color_image_from_npy(npy_path,save_path,img_w, img_h):
    #convert segmentation result to color image
    #[512,512,3],each element indicate probability of semantic class 0 to 100
    npy = np.load(npy_path)
    color = np.zeros((512,512,3))
    semantic_result = npy

    for semantic_ID in range(all_label_in_cityscapes):
        color_map = semantic_label_to_BGR[ID_to_semantic_label[semantic_ID]]
        color[np.where((semantic_result == semantic_ID))] = color_map
        
    color_resized = cv2.resize(color,(img_w,img_h))
    cv2.imwrite(save_path,color_resized)


date1 = "2015-10-30-13-52-14"
date2 = "2015-11-10-10-32-52"
date3 = "2015-11-12-13-27-51"
date4 = "2015-11-13-10-28-08"
date5 = "2014-11-25-09-18-32"
date6 = "2015-08-28-09-50-22"

    
#date_list = [date1,date2,date3,date4,date5,date6]

date_list = [date6]

for date in date_list:
    os.makedirs("result/" + date + "/segmentation_result_color/",exist_ok = True)
    for fname in tqdm(sorted(glob.glob("result/" + date + "/seg/*"))):
        output_fname = os.path.basename(fname).split(".")[0] + ".jpg"
        output_path = "result/" + date + "/segmentation_result_color/" + output_fname
        generate_semantic_color_image_from_npy(fname,output_path,img_w = 1080, img_h = 800)
        

    









