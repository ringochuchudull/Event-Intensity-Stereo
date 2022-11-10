import os
from PIL import Image
import torchvision
import cv2
import numpy as np
import torch
import pickle


def get_keys_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    keys = meta_info['keys']
    return keys

def get_metainfo_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    return meta_info
