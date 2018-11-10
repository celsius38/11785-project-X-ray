import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import os.path
from tqdm import tqdm

def get_id_label_list(): 
    """ @ Return: the entire image_id - label list with the same sequential order""" 
    data_entry = pd.read_csv("data/Data_Entry_2017.csv") 
    id_list = data_entry[["Image Index"]].values.flatten().tolist()
    labels = data_entry.apply(lambda x: x["Finding Labels"].split("|"), axis=1)
    mlb = MultiLabelBinarizer()  
    label_list = mlb.fit_transform(labels)
    return id_list, label_list

IMG_PATHs =["data/images" + str(i) for i in range(1,13)]
IMBALANCE_CUTOFF = 0.5

def save_npy():
    """
    for test data specified in the `test_list.txt`, generate a single npy
     file that contains the data and corresponding label in sequential order
    for train data specified in the `train_val.txt`, split into several reasonable 
     sized npy file that contains the data and corresponding label in sequential order
    @Note:
      labels are encoded using one-hot-bit encoding
      images are concatenated into N x C x W x H (N is number of images in a single set, 
      				C is 1 as black-white, W for width and H for height)
    """ 
    id_list, label_list = get_id_label_list()
    id_idx = 0      # running ptr in the id_list
    dir_idx = 0     # ptr to the directory
    alldata, alllabel = [], []

    with open("data/test_list.txt", "r") as test_list:
        for test_id in tqdm(test_list):
            test_id = test_id.strip()
            # loop through the index list to find the matching idx
            while id_list[id_idx] != test_id : 
                id_idx += 1   
            path = os.path.join(IMG_PATHs[dir_idx], test_id)
            if not os.path.isfile(path):
                dir_idx += 1
                path = os.path.join(IMG_PATHs[dir_idx], test_id)
                assert(os.path.isfile(path))

            data = cv2.imread(path, 0)
            label = label_list[id_idx] 
            alldata.append(data)
            alllabel.append(label)
    print("finish reading, now saving to npy...")
    alldata = np.array(alldata)
    alllabel = np.array(alllabel)
    np.save("test.npy", alldata)
    np.save("test_label.npy", alllabel)
    print("Done!")

             

        
             
            
            

    
    










