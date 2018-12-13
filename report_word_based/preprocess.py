from xml.dom import minidom
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os; from os import listdir
import json
from tqdm import tqdm
from natsort import natsorted, ns

def dump_json(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f)

def load_json(file):
    with open(file, "rb") as f:
        obj = json.load(f)
    return obj

WORD_TO_INT = load_json("data/WORD_TO_INT.js")
INT_TO_WORD = load_json("data/INT_TO_WORD.js")
INT_TO_WORD = {int(k): v for k ,v in INT_TO_WORD.items()}

def str_to_int(s):
    # given a string, transcribe each word to int and return int list
    s = s.lower().split()
    res = [WORD_TO_INT[c] for c in s]
    return res

def int_to_str(x):
    # given an int numpy array, return string that corresponds to it
    res = np.vectorize(INT_TO_WORD.get)(x)
    res = res.tolist()
    return " ".join(res)

def append_file(path, *args):
    with open(path, "a") as f:
        for arg in args[:-1]:
            f.write(str(arg) + ",")
        f.write(str(args[-1]))
        f.write("\n")

def target_extractor(path): 
    """
    @Param:
        path: dir that contains all reports
    @Return:
        targets: xml_name -> triplet (indication, findings, images)
        mapping: img_name -> xml_name
    """
    def _target_extractor(xml):
        doc = minidom.parse(xml)
        # fetch indication, findings and impression
        abstracts = doc.getElementsByTagName("AbstractText")
        gettext = lambda ele:ele.firstChild.nodeValue if ele.firstChild else ""
        _, indication, findings, impression = [gettext(ele) for ele in abstracts]
        # fetch target images
        images = doc.getElementsByTagName("parentImage")
        getid   = lambda ele: ele.getAttribute("id")
        images  = [getid(img) for img in images]
        return (indication, findings, impression), images 

    targets = {}
    mapping = {}
    for xml in tqdm(natsorted(listdir(path), alg=ns.IGNORECASE)):
        target, image = _target_extractor(os.path.join(path, xml))
        targets[xml] = target
        mapping.update({img : xml for img in image})
    return targets, mapping

def image_preprocess(path):
    """
    @Param:
        path: dir that contains all images
    @Return:
        img_names:  list of image names
        img_list:   np.array of images in the same order as img_names
    """
    def _image_preprocess(img_path):
        img = cv2.imread(img_path, 0) 
        if img is None:
            return None
        W,H = img.shape
        # truncate to (512, 512)
        W, H = min(512, W), min(512, H)
        img = img[:W, :H]
        # pad to (512, 512) 
        if(W < 512 or H < 512):
            img = np.pad(img, ((0, 512-W), (0, 512-H)), "constant", constant_values = 0)
        return img
    img_names    = []
    img_list     = []
      
    for img in tqdm(natsorted(listdir(path), alg=ns.IGNORECASE)):
        img_path = os.path.join(path, img) 
        parsed_img = _image_preprocess(img_path)
        if parsed_img is not None:
            img_names.append(img)
            img_list.append(parsed_img) 
    return img_names, np.array(img_list)

def transcribe():
    """
    transcribe indications, findings, and impressions into numpy int array,
    in the same sequential order and same length as images
    """
    img_list, mapping, targets = (  load_json("data/img_names.js"), 
                                    load_json("data/mapping.js"), 
                                    load_json("data/targets.js") )
    indications = []
    findings = []
    impressions = []

    for img in img_list:
        xml = mapping[img]
        target = targets[xml] 
        indications.append(target[0])
        findings.append(target[1])
        impressions.append(target[2])
    indications = np.array([np.array(str_to_int(i)) for i in indications])
    findings    = np.array([np.array(str_to_int(f)) for f in findings])
    impressions = np.array([np.array(str_to_int(i)) for i in impressions])

    return indications, findings, impressions