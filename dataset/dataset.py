import glob
import os
import cv2
import lmdb
from argparse import ArgumentParser

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from .data_utils import *


class EventBinslmdb(Dataset):
    def __init__(self,
                 lmdb_path,
                 h=180,
                 w=240,
                 num_bins=64):
        super(EventBinslmdb, self).__init__()
        self.env = lmdb.open(lmdb_path,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.keys = get_keys_from_lmdb(lmdb_path)
        self.h = h
        self.w = w
        self.num_bins = num_bins

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lmdb_path', type=str, default='/home/mist/Data/ETHDVS_E_0.15s64nb.lmdb')
        parser.add_argument('--dvs_data_num_bins', type=int, default=64)
        parser.add_argument('--dvs_data_w', type=int, default=240)
        parser.add_argument('--dvs_data_h', type=int, default=180)
        return parser

    @staticmethod
    def from_namespace(args):
        instance = EventBinslmdb(
            lmdb_path=args.lmdb_path,
            h=args.dvs_data_h,
            w=args.dvs_data_w,
            num_bins=args.dvs_data_num_bins
        )
        return instance

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        event_bins = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return np.clip(event_bins.reshape((2, self.num_bins, self.h, self.w)), 0., 1.)  #, np.arange(self.num_bins).astype(np.float32) / self.num_bins


class StereoDVS_woE_lmdb(Dataset):
    def __init__(self,
                 lmdb_path,
                 recon_type=0):
        super(StereoDVS_woE_lmdb, self).__init__()
        self.env = lmdb.open(lmdb_path,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.metainfo = get_metainfo_from_lmdb(lmdb_path)
        self.size = self.metainfo['size']
        self.keys = self.metainfo['keys']
        self.recon_type = recon_type

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        image_stack = np.copy(np.frombuffer(buf, dtype=np.float32))
        image_stack = image_stack.reshape((6, 1) + self.size)
        left_int = image_stack[0] / 255.
        right_int = image_stack[1] / 255.
        left_dis = image_stack[2]
        right_dis = image_stack[3]
        right_recon = image_stack[4] / 255.
        right_recon2 = image_stack[5] / 255.
        if self.recon_type == 0:
            return left_int, right_int, left_dis, right_dis, right_recon
        elif self.recon_type == 1:
            return left_int, right_int, left_dis, right_dis, right_recon2
        else:
            return left_int, right_int, left_dis, right_dis, right_recon
