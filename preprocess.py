import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import os.path
from tqdm import tqdm

IMG_PATHs =["data/images" + str(i) for i in range(1,13)] 
MAX_SIZE = 10000

def get_id_label_list(): 
    """ @ Return: the entire image_id - label list(using one-hot-encoding) 
                    with the same sequential order,
                    also return the mapping rule that one-hot-encoding uses
    """ 
    data_entry = pd.read_csv("data/Data_Entry_2017.csv") 
    id_list = data_entry[["Image Index"]].values.flatten().tolist()
    labels = data_entry.apply(lambda x: x["Finding Labels"].split("|"), axis=1)
    mlb = MultiLabelBinarizer()  
    label_list = mlb.fit_transform(labels)
    mapping_list = list(mlb.classes_) # fetch the mapping rule used in the binarizer
    return id_list, label_list, mapping_list

def train_val_split():
    """Split the 'data/train_val_list.txt` into two lists: train_list.npy and val_list.npy"""
    arr = []
    with open("data/train_val_list.txt", "r") as f:
        prev_identity = None
        for i in f:
            i = i.strip()
            identity, index = i.split('_')
            if identity != prev_identity:
                arr.append([])
                prev_identity = identity
            arr[-1].append(i)
    train_arr = []
    val_arr = []
    for a in arr:
        if np.random.rand() < 0.1 :
            val_arr.append(a)
        else:
            train_arr.append(a)

    with open("train_list.txt", "w") as f:
        for idx in chain(*train_arr):
            f.write(idx + "\n")
    with open("val_list.txt", "w") as f:
        for idx in chain(*val_arr):
            f.write(idx + "\n")

def save_npy(mode, skip = 0):
    """ 
    for test data specified in the `test_list.txt`, generate a single npy
     file that contains the data and corresponding label in sequential order
    for train data specified in the `train_val.txt`, split into several reasonable 
     sized npy file that contains the data and corresponding label in sequential order
    @Param: 
        mode: "train", "val" or "test"
        skip: skip the first skip number of chunks
    @Note:
      labels are encoded using one-hot-bit encoding
      images are concatenated into N x C x W x H (N is number of images in a single set, 
      				C is 1 as black-white, W for width and H for height)
    """ 
    def save_procedure(data, label, file_cnt):
        """data and label are nparray, file_cnt specify the index """
        nonlocal mode
        data_path = "{}{}.npy".format(mode,file_cnt) 
        label_path = "{}_label{}.npy".format(mode, file_cnt)
        np.save(data_path, data)
        np.save(label_path, label)
        print("saved to {} & {}".format(data_path, label_path))

    id_list, label_list = get_id_label_list()
    id_idx = 0      # running ptr in the id_list
    dir_idx = 0     # ptr to the directory
    alldata, alllabel = [], []

    cnt = 0; file_cnt = 0
    with open("data/{}_list.txt".format(mode), "r") as lst:
        for tid in tqdm(lst):
            tid = tid.strip()
            # loop through the index list to find the matching idx 
            while id_list[id_idx] != tid:
                id_idx += 1   
            path = os.path.join(IMG_PATHs[dir_idx], tid)
            while not os.path.isfile(path):
                dir_idx += 1
                path = os.path.join(IMG_PATHs[dir_idx], tid)

            data = cv2.imread(path, 0)
            label = label_list[id_idx] 
            alldata.append(data)
            alllabel.append(label)
            id_idx += 1 
            cnt += 1

            if(cnt == MAX_SIZE):
                print("reaching limit, now saving to npy...")
                alldata = np.array(alldata)
                alllabel = np.array(alllabel) 
                if(file_cnt < skip):
                    print("[{}/{}] skipped saving...".format(file_cnt + 1, skip))
                else:
                    save_procedure(alldata, alllabel, file_cnt) 
                    print("done")
                cnt = 0
                file_cnt += 1 
                alldata = []
                alllabel = []
        print("finalize, now saving the rest {} to npy...".format(len(alldata)))
        alldata = np.array(alldata)
        alllabel = np.array(alllabel)
        save_procedure(alldata, alllabel, file_cnt)
        print("done")
     

             

        
             
            
            

    
    










