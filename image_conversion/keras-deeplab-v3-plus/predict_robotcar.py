import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from model import Deeplabv3
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array,img_to_array, array_to_img
import glob
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append("../../paramdir")
import parameters
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

def load_image_deeplab(path,to_dir):
    mean_subtraction_value=127.5
    loaded_image=load_img(path, grayscale=False, color_mode='rgb',target_size = (512,512))
    loaded_arr=img_to_array(loaded_image)
    completed_image=(loaded_arr / mean_subtraction_value) - 1.
    return completed_image


#image = np.array(Image.open('1446121603259779.png'))
#path='1446121603259779.png'



deeplab_model = Deeplabv3(weights="cityscapes" ,classes=19,backbone = 'xception',)


date1 = parameters.train_date
date2 = parameters.test_date

for date in [date1,date2]:
    print(date)
    os.makedirs("result/"+date+"/seg",exist_ok = True)
    pbar=tqdm(total=len(glob.glob("../../raw_image/"+date+"/*")))
    for path in sorted(glob.glob("../../raw_image/"+date+"/*")):
        loaded_image=load_image_deeplab(path,"result/"+date+"/resized")
        # make prediction
        #print("resized_image_shape",resized_image.shape)
        res = deeplab_model.predict(np.expand_dims(loaded_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        np.save("result/"+date+"/seg/"+path.split("/")[-1].split(".")[0],labels)
        pbar.update(1)
    pbar.close()
    """
    labels_img=np.reshape(labels,(512,512,1))
    labels_img=array_to_img(labels_img)
    labels_img.save("laa.png")
    """




#plt.imshow(labels)
#plt.savefig('figure.png')
#plt.waitforbuttonpress()

