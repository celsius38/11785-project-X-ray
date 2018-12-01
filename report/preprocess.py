from xml.dom import minidom
import torch
import numpy as np


def target_extractor(path): 
    """
    @Param:
        path: dir that contains all reports
    @Side:
        write single npz file that contains all reports 
        and corresponding image paths
    """
    def _target_extractor(xml):
        doc = minidom.parse(xml)
        # fetch indication, findings and impression
        abstracts = doc.getElementsByTagName("AbstractText")
        gettext = lambda ele: ele.firstChild.nodeValue
        _, indication, findings, impression = [gettext(ele) for ele in abstracts]
        # fetch target images
        images = doc.getElementsByTagName("parentImage")
        getid   = lambda ele: ele.getAttribute("id")
        images  = [getid(img) for img in images]
        return (indication, findings, impression), images 
    


            

