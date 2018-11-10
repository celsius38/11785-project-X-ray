import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def get_label_list():
    data_entry = pd.read_csv("data/Data_Entry_2017.csv")
    labels = data_entry.apply(lambda x: x["Finding Labels"].split("|"), axis=1)
    mlb = MultiLabelBinarizer() 
    label_list = mlb.fit_transform(labels)
    # label_list_df = pd.DataFrame(data=label_list, columns=mlb.classes_)
    return label_list

  # filename = [zip file names]

IMG_PATHs =["data/images1/","data/images2/","data/images3/","data/images4/","data/images5/"]
start_index = 0
imbalance_cutoff = 0.5

def data_preprocess(IMG_PATH,label_list):
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
    # TODO:

    # get the length
    global start_index, imbalance_cutoff

    file_names = glob.glob(IMG_PATH + '*.png')

    num_image = len(file_names)

    # check label distribution
    cur_label_list = label_list[start_index:start_index+num_image]

    if (np.sum(cur_label_list,axis=0)/num_image)[10] > imbalance_cutoff:
        label_imbalance = True
    else:
        label_imbalance = False
      
    images = []
    for f in file_names:
        im = cv2.imread(f, 0)
        images.append(im)
    image_reshape = np.expand_dims(np.array(images), axis=1)
    # image_reshape.shape

    return (image_reshape,cur_label_list),label_imbalance











