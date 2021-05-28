import numpy as np
import pickle
from tqdm import tqdm
import glob
import os
import sys

sys.path.append("../../paramdir")
import parameters

args = sys.argv



#gps_dict = {i[0]:np.array([float(i[1]),float(i[2])]) for i in gps_str}
#print(gps_dict)


#https://neuryo.hatenablog.com/entry/2019/02/06/111611
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)



date_list = [parameters.train_date, parameters.test_date]




#image_conversions = ["canny","semantic","depth"]
image_conversions = ["Raw", "canny","semantic"]

for ic in image_conversions:
    for date in date_list:
        os.makedirs('dict',exist_ok=True)
        print(date,ic)
        image_to_NetVLAD_test = {}
        for path in tqdm(sorted(glob.glob("netvlad_feature_" + ic + "/" + date + "/*"))):
            #print(semantic_npy_path)
            png_name = os.path.basename(path)
            name = png_name.split(".")[0]
            image_to_NetVLAD_test[name] = np.loadtxt(path)
        pickle_dump(image_to_NetVLAD_test, "dict/"+date+'_image_to_' + ic + '_netvlad_pred_full_size.pickle')
