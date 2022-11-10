import os
import sys
import torch
import torchvision.transforms.functional as tF

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.notebook_utils import *
import lmdb
import pickle


data_name = 'ETHDVS_E_0.05s'
ETH_data_root = '/home/mist/Data/ETH_DVS/'
event_txt_name = 'events.txt'
image_txt_name = 'images.txt'
image_path_name = 'images'
lmdb_saving_path = f'/home/mist/Data/{data_name}.lmdb'
if os.path.exists(lmdb_saving_path):
    print(f'Folder {lmdb_saving_path} already exists. Exit...')

event_slice_time = 0.05  # in second
commit_time = 1000
num_bins = 64
max_w = 240; max_h = 180

data_name_nb = f'{data_name}64nb'
lmdb_saving_path_nb = f'/home/mist/Data/{data_name}64nb.lmdb'
if os.path.exists(lmdb_saving_path_nb):
    raise Exception(f'Folder {lmdb_saving_path_nb} already exists. Exit...')


#### create lmdb environment
env = lmdb.open(lmdb_saving_path_nb, map_size=1099511627776)


#### write data to lmdb
video_list = os.listdir(ETH_data_root)
txn = env.begin(write=True)
event_keys = []
commit_count = 0
for i, video in enumerate(video_list):
    print(f'Processing {i}th video {video}...')
    event_path = os.path.join(ETH_data_root, video, event_txt_name)
    with open(event_path, 'r') as ef:
        line = ef.readline()
        current_events = []
        while line:
            t, x, y, p = parsing_event_txt_line(line)
            if len(current_events) == 0:
                current_events.append(np.array([t, x, y, p]))
            else:
                if t < (current_events[0][0] + event_slice_time):
                    current_events.append(np.array([t, x, y, p]))
                else:
                    current_event_bins = events2bins(current_events, max_w, max_h, num_bins)[0].numpy().astype(np.uint8)
                    key = f'{video}:{current_events[0][0]}-{current_events[-1][0]}'
                    key_byte = key.encode('ascii')
                    print(f'{commit_count}:{key}')
                    txn.put(key_byte, current_event_bins)
                    event_keys.append(key)
                    commit_count += 1
                    if commit_count % commit_time == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                        print('Commit lmdb...')
                        meta_info = {'name': data_name_nb}
                        meta_info['keys'] = event_keys
                        pickle.dump(meta_info, open(os.path.join(lmdb_saving_path_nb, 'meta_info.pkl'), 'wb'))
                        print('Finish writing meta info.')
                    current_events = [np.array([t, x, y, p])]
            line = ef.readline()

txn.commit()
env.close()
print('Finish writing lmdb.')

#### write meta indormation
meta_info = {'name': data_name_nb}
meta_info['keys'] = event_keys
pickle.dump(meta_info, open(os.path.join(lmdb_saving_path_nb, 'meta_info.pkl'), 'wb'))
print('Finish writing meta info.')



