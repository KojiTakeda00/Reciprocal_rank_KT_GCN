import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import sys

args = sys.argv


date1 = "2015-08-28-09-50-22"
date2 = "2015-10-30-13-52-14"

"""
for date in [date1,date2]:
    for pic in tqdm(sorted(glob.glob("/media/chinourobot/HDCL-UT1/takeda/keras-deeplab-v3-plus/result/" + date + "/resized/*"))):
        os.makedirs(date,exist_ok = True)
        img = cv2.imread(pic,0)
        edges = cv2.Canny(img,100,200)
        name = os.path.basename(pic)
        cv2.imwrite(date + "/" + name,edges)
"""


for date in [date1,date2]:
    for pic in tqdm(sorted(glob.glob("../" + date + "/all/*"))):
        #print(pic)
        os.makedirs("canny_edge_img/" + date,exist_ok = True)
        img = cv2.imread(pic,0)
        edges = cv2.Canny(img,100,200)
        name = os.path.basename(pic)
        cv2.imwrite("canny_edge_img/" + date + "/" + name,edges)
