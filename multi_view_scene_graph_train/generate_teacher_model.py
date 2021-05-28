# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys
#from keras.utils import np_utils
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import os
#from keras.preprocessing.image import load_img, img_to_array, array_to_img
#from keras.utils.np_utils import to_categorical
from PIL import Image
import numpy as np
import pandas as pd
import re
import shutil
#from keras.utils import plot_model
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("../paramdir")
import parameters

args=sys.argv

#date_train = "2015-10-30-13-52-14"
#date_test = "2015-11-13-10-28-08"

date_train = parameters.train_date
date_test = parameters.test_date

class_list_train = np.loadtxt("../class_list/train_" + date_train +"_test_" + date_test + "/class_list_train_"+date_train+".txt",str)
class_list_test = np.loadtxt("../class_list/train_" + date_train + "_test_" + date_test + "/class_list_test_"+date_test +".txt",str)


###generate test matrix
bf = cv2.BFMatcher()

        
"""
for test_file in class_list_SUMMER[:,0]:
   test_txt = test_file.split(".")[0] + ".txt"
   test_feature = np.loadtxt("../../../../netvlad_tf_open/python/extracted_data/robotcar/" + date_test + "/" + test_txt)
   test_features_list.append(test_feature)

test_features_array = np.array(test_features_list,np.float32)
"""


import pickle
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


###generate test matrix
bf = cv2.BFMatcher()

#2015-08-28-09-50-22_image_to_RGB_netvlad_pred_full_size.pickle

for image_conversion in ["Raw","semantic","canny"]:

    ##modify
    test_netvlad_dict = pickle_load("../netvlad_tf_open/python/dict/" + date_test + "_image_to_" + image_conversion + "_netvlad_pred_full_size.pickle")
    train_netvlad_dict = pickle_load("../netvlad_tf_open/python/dict/" + date_train + "_image_to_" + image_conversion + "_netvlad_pred_full_size.pickle")


    test_features_list = []

    for test_file in tqdm(class_list_test[:,0]):
        test_feature = test_netvlad_dict[test_file.split(".")[0]]
        test_features_list.append(test_feature)

    test_features_array = np.array(test_features_list,np.float32)

    print(test_features_array[0][0])
    print(class_list_test[0])
    train_features_list = []

    for train_file in tqdm(class_list_train[:,0]):
        train_feature = train_netvlad_dict[train_file.split(".")[0]]
        train_features_list.append(train_feature)

    train_features_array = np.array(train_features_list,np.float32)

    print(train_features_array.shape)
    print(test_features_array.shape)

    ####generate train dictt
    all_category_num = len(set(class_list_train[:,1]))

    print(all_category_num)


    class_list_database = class_list_train


    test_result = []
    for i in tqdm(range(all_category_num)):
        database_features_list = []
        for j in class_list_database:
            if j[1] == str(i):
                feature = train_netvlad_dict[j[0].split(".")[0]]
                database_features_list.append(feature)
        database_features_array = np.array(database_features_list,np.float32)
        #print(database_features_array.shape)
        matches = bf.knnMatch(test_features_array,database_features_array, k=1)
        match_distance = [match[0].distance for match in matches]
        test_result.append(match_distance)
    np.savetxt("test_result_" + image_conversion + ".txt",np.array(test_result))
    result = np.argmin(np.array(test_result).T,axis = 1)

    GT = class_list_test[:,1].astype(np.uint8)
    print(result == GT)
    print(result)
    print(np.sum(result == GT) / len(GT))




    train_result = []
    for i in tqdm(range(all_category_num)):
        database_features_list = []
        for j in class_list_database:
            if j[1] == str(i):
                feature = train_netvlad_dict[j[0].split(".")[0]]
                database_features_list.append(feature)
        database_features_array = np.array(database_features_list,np.float32)
        matches = bf.knnMatch(train_features_array,database_features_array, k=1)
        match_distance = [match[0].distance for match in matches]
        train_result.append(match_distance)
    np.savetxt("train_result_" + image_conversion + ".txt",np.array(train_result))
    result = np.argmin(np.array(train_result).T,axis = 1)

    GT = class_list_train[:,1].astype(np.uint8)
    print(result == GT)
    print(result)
    print(np.sum(result == GT) / len(GT))



