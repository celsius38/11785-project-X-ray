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

def str_to_int(s):
    # given a string, turn to lower case and return int list
    s = s.lower()
    res = [CHAR_TO_INT[c] for c in s]
    return res

def int_to_str(x):
    # given an int numpy array, return string that corresponds to it
    res = np.vectorize(INT_TO_CHAR.get)(x)
    res = res.tolist()
    return "".join(res)

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

def dump_json(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f)

def load_json(file):
    with open(file, "rb") as f:
        obj = json.load(f)
    return obj

def get_char_set():
    targets = load_json("data/targets.js")
    sets = [set(v[0]) | set(v[1]) | set(v[2]) for v in targets.values()]
    char_set = set.union(*sets)
    char_set = {c.lower() for c in char_set}
    return sorted(list(char_set))

def transcribe():
    """
    transcribe indications, findings, and impressions into numpy int array,
    in the same sequential order and same length as images
    """
    img_list, mapping, targets = (  load_json("data/img_list.js"), 
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

INT_TO_CHAR =  {0: '', # <SOS>/<EOS>
 1: ' ',
 2: '"',
 3: '#',
 4: '%',
 5: '&',
 6: "'",
 7: '(',
 8: ')',
 9: '+',
 10: ',',
 11: '-',
 12: '.',
 13: '/',
 14: '0',
 15: '1',
 16: '2',
 17: '3',
 18: '4',
 19: '5',
 20: '6',
 21: '7',
 22: '8',
 23: '9',
 24: ':',
 25: ';',
 26: '<',
 27: '>',
 28: '?',
 29: '[',
 30: '\\',
 31: ']',
 32: 'a',
 33: 'b',
 34: 'c',
 35: 'd',
 36: 'e',
 37: 'f',
 38: 'g',
 39: 'h',
 40: 'i',
 41: 'j',
 42: 'k',
 43: 'l',
 44: 'm',
 45: 'n',
 46: 'o',
 47: 'p',
 48: 'q',
 49: 'r',
 50: 's',
 51: 't',
 52: 'u',
 53: 'v',
 54: 'w',
 55: 'x',
 56: 'y',
 57: 'z'}

CHAR_TO_INT = {'': 0, # <SOS>/<EOS>
 ' ': 1,
 '"': 2,
 '#': 3,
 '%': 4,
 '&': 5,
 "'": 6,
 '(': 7,
 ')': 8,
 '+': 9,
 ',': 10,
 '-': 11,
 '.': 12,
 '/': 13,
 '0': 14,
 '1': 15,
 '2': 16,
 '3': 17,
 '4': 18,
 '5': 19,
 '6': 20,
 '7': 21,
 '8': 22,
 '9': 23,
 ':': 24,
 ';': 25,
 '<': 26,
 '>': 27,
 '?': 28,
 '[': 29,
 '\\': 30,
 ']': 31,
 'a': 32,
 'b': 33,
 'c': 34,
 'd': 35,
 'e': 36,
 'f': 37,
 'g': 38,
 'h': 39,
 'i': 40,
 'j': 41,
 'k': 42,
 'l': 43,
 'm': 44,
 'n': 45,
 'o': 46,
 'p': 47,
 'q': 48,
 'r': 49,
 's': 50,
 't': 51,
 'u': 52,
 'v': 53,
 'w': 54,
 'x': 55,
 'y': 56,
 'z': 57
}    
