import torch
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from dataset.dataset import EventBinslmdb, StereoDVS_woE_lmdb


class EventBinsLoader(pl.LightningDataModule):
    def __init__(self,
                 lmdb_path,
                 no_shuffle=False,
                 num_workers=8,
                 batch_size=32,
                 num_valid_samples=256,
                 ):
        super(EventBinsLoader, self).__init__()
        self.lmdb_path = lmdb_path
        self.no_shuffle = no_shuffle
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_valid_samples = num_valid_samples

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Training dataset type
        parser.add_argument('--lmdb_path', type=str, default='')
        # args for dataloader
        parser.add_argument('--no_shuffle', action='store_false', help='--no_shuffle for no shuffle')
        parser.add_argument('--num_workers', type=int, default=12)
        parser.add_argument('--batch_size', type=int, default=32)
        # validation dataset
        parser.add_argument('--num_valid_samples', type=int, default=128)
        return parser

    @staticmethod
    def from_namespace(args):
        instance = EventBinsLoader(
            lmdb_path=args.lmdb_path,
            no_shuffle=args.no_shuffle,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            num_valid_samples=args.num_valid_samples
        )
        return instance

    def setup(self, stage=None):
        self.dataset = EventBinslmdb(lmdb_path=self.lmdb_path)
        self.train, self.validation = random_split(self.dataset, [len(self.dataset) - self.num_valid_samples, self.num_valid_samples])

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.no_shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


#here
class StereoDVS_woE_Loader(pl.LightningDataModule):
    def __init__(self,
                 lmdb_path,
                 validation_lmdb_path,
                 no_shuffle=False,
                 recon_type=0,
                 num_workers=8,
                 batch_size=1,
                 num_valid_samples=256):
        super(StereoDVS_woE_Loader, self).__init__()
        self.lmdb_path = lmdb_path
        self.validation_lmdb_path = validation_lmdb_path
        self.recon_type = recon_type
        self.no_shuffle = no_shuffle
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_valid_samples = num_valid_samples

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Training dataset type
        parser.add_argument('--lmdb_path', type=str, default='')
        parser.add_argument('--validation_lmdb_path', type=str, default='')
        # args for dataloader
        parser.add_argument('--no_shuffle', action='store_false', help='--no_shuffle for no shuffle')
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--recon_type', type=int, default=0, help='`0` for firenet, and `1` for e2vid')
        parser.add_argument('--batch_size', type=int, default=8)
        # validation dataset
        parser.add_argument('--num_valid_samples', type=int, default=128)
        return parser

    @staticmethod
    def from_namespace(args):
        instance = StereoDVS_woE_Loader(
            lmdb_path=args.lmdb_path,
            validation_lmdb_path=args.validation_lmdb_path,
            recon_type=args.recon_type,
            no_shuffle=args.no_shuffle,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            num_valid_samples=args.num_valid_samples
        )
        return instance

    def setup(self, stage=None):
        dataset = StereoDVS_woE_lmdb(lmdb_path=self.lmdb_path)
        if os.path.isdir(self.validation_lmdb_path):
            self.train = dataset
            self.validation = StereoDVS_woE_lmdb(lmdb_path=self.validation_lmdb_path)
        else:
            self.train, self.validation = random_split(dataset, [len(dataset) - self.num_valid_samples, self.num_valid_samples])

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.no_shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )