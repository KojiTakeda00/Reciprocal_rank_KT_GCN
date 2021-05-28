import numpy as np
import pickle




date1 = "2015-10-30-13-52-14"
date2 = "2015-08-28-09-50-22"

#https://neuryo.hatenablog.com/entry/2019/02/06/111611
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)


for date in [date1,date2]:
    gps_str = np.loadtxt("/mnt/hdd1/takeda/RobotCarDataset-Scraper/Downloads/"+date+"/interporated_gps.txt",str)
    gps_dict = {i[0]:np.array([float(i[1]),float(i[2])]) for i in gps_str}

    pickle_dump(gps_dict, "gps_dict/" + date+'_gps.pickle')
