import os
import sys
import torch
import torchvision.transforms.functional as tF
import cv2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.notebook_utils import *
from utils.file_utils import parse_image_folder_for_stereo_DVS, parse_folder_for_stereo_DVS
import lmdb
import pickle


data_name = 'DVS_Stereo_Blur_Test'
data_root = '/mnt/lustrenew/share_data/gujinjin1/Stereo-Blur-Dataset-Test'
left_intensity_pathname = 'gray_left'
right_intensity_pathname = 'gray_right'
right_reconstruct_pathname = 'recons_FireNet'
right_reconstruct_pathname2 = 'recons_E2VID'
right_disparity_pathname = 'disparity_right'
left_disparity_pathname = 'disparity_left'
event_txt = 'event_right_raw/events.txt'

lmdb_saving_path = f'/mnt/lustrenew/share_data/gujinjin1/{data_name}.lmdb'
if os.path.exists(lmdb_saving_path):
    print(f'Folder {lmdb_saving_path} already exists. Exit...')
    sys.exit(1)

#### create lmdb environment
env = lmdb.open(lmdb_saving_path, map_size=1099511627776)
txn = env.begin(write=True)
video_list = os.listdir(data_root)
data_keys = []
commit_count = 0
commit_time = 1000
for i, video in enumerate(video_list):
    print(f'Processing {i}th video {video}...')
    left_intensities = parse_image_folder_for_stereo_DVS(os.path.join(data_root, video, left_intensity_pathname))
    right_intensities = parse_image_folder_for_stereo_DVS(os.path.join(data_root, video, right_intensity_pathname))
    left_disparities = parse_folder_for_stereo_DVS(os.path.join(data_root, video, left_disparity_pathname))
    right_disparities = parse_folder_for_stereo_DVS(os.path.join(data_root, video, right_disparity_pathname))
    right_reconstructions = parse_image_folder_for_stereo_DVS(os.path.join(data_root, video, right_reconstruct_pathname))
    right_reconstructions2 = parse_image_folder_for_stereo_DVS(os.path.join(data_root, video, right_reconstruct_pathname2))
    event_txt_of_video = os.path.join(data_root, event_txt)

    if not \
        (len(left_intensities) == len(right_intensities)) and \
        (len(left_disparities) == len(right_disparities)) and \
        (len(left_intensities) == len(left_disparities)) and \
        (len(right_reconstructions) == len(right_intensities) - 1) and \
        (len(right_reconstructions2) == len(right_intensities) - 1):
        print(f'Wrong file numbers for video {video},'
              f'left_intensities: {len(left_intensities)}'
              f'right_intensities: {len(right_intensities)}'
              f'left_disparities: {len(left_disparities)}'
              f'right_disparities: {len(right_disparities)}'
              f'right_reconstructions: {len(right_reconstructions)}'
              f'right_reconstructions2: {len(right_reconstructions2)}')
        continue

    for j in range(20, len(right_intensities)):
        left_int = cv2.imread(os.path.join(data_root, video, left_intensity_pathname, left_intensities[j]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        right_int = cv2.imread(os.path.join(data_root, video, right_intensity_pathname, right_intensities[j]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        left_dis = np.load(os.path.join(data_root, video, left_disparity_pathname, left_disparities[j])).astype(np.float32)
        right_dis = np.load(os.path.join(data_root, video, right_disparity_pathname, right_disparities[j])).astype(np.float32)
        right_recon = cv2.imread(os.path.join(data_root, video, right_reconstruct_pathname, right_reconstructions[j - 1]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        right_recon = right_recon[:, :, 0] if right_recon.ndim == 3 else right_recon
        right_recon2 = cv2.imread(os.path.join(data_root, video, right_reconstruct_pathname2, right_reconstructions2[j - 1]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        right_recon2 = right_recon2[:, :, 0] if right_recon2.ndim == 3 else right_recon2
        key = f'{video}/{j}'
        key_byte = key.encode('ascii')
        print(f'{commit_count}:{key}')
        data_pkg = np.stack([left_int, right_int, left_dis, right_dis, right_recon, right_recon2], axis=0)
        txn.put(key_byte, data_pkg)
        data_keys.append(key)
        commit_count += 1
        if commit_count % commit_time == 0:
            txn.commit()
            txn = env.begin(write=True)
            print('Commit lmdb...')
            meta_info = {'name': data_name}
            meta_info['keys'] = data_keys
            meta_info['size'] = data_pkg.shape[-2:]
            pickle.dump(meta_info, open(os.path.join(lmdb_saving_path, 'meta_info.pkl'), 'wb'))
            print('Finish writing meta info.')

txn.commit()
env.close()
print('Finish writing lmdb.')
meta_info = {'name': data_name}
meta_info['keys'] = data_keys
meta_info['size'] = data_pkg.shape[-2:]
pickle.dump(meta_info, open(os.path.join(lmdb_saving_path, 'meta_info.pkl'), 'wb'))
print('Finish writing meta info.')



