import numpy as np
import glob

for elem in sorted(glob.glob("train*")):
    train_list_path = glob.glob(elem + "/class_list_train*")[0]
    #test_list_path =  glob.glob(elem + "/class_list_test*")[0]
    class_train = np.loadtxt(train_list_path,str)[:,1]
    #class_test = np.loadtxt(test_list_path)[:,1]
    print(elem)
    print(int(len(set(class_train))))
