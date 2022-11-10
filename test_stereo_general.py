from argparse import ArgumentParser
import warnings

import torch
import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models import load_model
from models.general_ssl_stereo import EventStereoSelfSuperviseGeneral
from models import GeneralStereoLoggingCallback
from networks import load_network
from dataset.dataloader import StereoDVS_woE_Loader
from utils.trainer_args import StereoDVS_Trainer_Default


warnings.filterwarnings('ignore')


def main(args):
    # ================================ Initializing
    # init network and model
    model = EventStereoSelfSuperviseGeneral.from_namespace(args).load_from_checkpoint(args.checkpoint_file)

    # init dataloader
    pl_eventbins_data = StereoDVS_woE_Loader.from_namespace(args)
    pl_eventbins_data.setup()


    val_dataloader = pl_eventbins_data.val_dataloader()

    print(type(val_dataloader))

    val_dataloader = val_dataloader

    print(len(val_dataloader))

    # init logger
    logger = TensorBoardLogger(os.path.join(args.logger_dir, args.exp_name), name=args.model_type)

    # model saving
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.default_root_dir,
        verbose=True,
        save_last=True,
        # prefix=args.exp_name, TODO: I don't know why this version does not have this param
        period=args.check_val_every_n_epoch,
        # this is to save model to `last`, in callback, we save model in separate files
        filename='{epoch}-{step}',
    )

    # init trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=args.gpu_useage,
                                            precision=args.training_precision,
                                            logger=logger,
                                            callbacks=[model_checkpoint_callback, GeneralStereoLoggingCallback()])

    # ================================ Train
    #trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    #trainer.test(model, test_dataloaders=model.val_dataloader())

    print('run test')
    trainer.test(model, test_dataloaders=val_dataloader)



if __name__ == "__main__":
    # fix the seed for reproducing
    pl.seed_everything(1)

    #================================ ArgParsing
    # init Argment Parser
    parser = ArgumentParser()
    # add all the available trainer options to argparse, Check trainer's paras for help
    parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument('--model_type', type=str, default='general_ssl_stereo', help='stereo_ss_bsl')
    parser.add_argument('--logger_dir', type=str, default='./EXPs/tb_logs_general_test', help='logging path')
    parser.add_argument('--gpu_useage', type=int, default=1, help='Over write --gpus, it does not work now')
    parser.add_argument('--training_precision', type=int, default=32, help='')
    parser.add_argument('--checkpoint_file', type=str, default='./pretrained_models/model_weight.ckpt', help='Path to checkpoint file')

    temp_args, _ = parser.parse_known_args()

    # add model specific args
    Stereo_Model = load_model(temp_args.model_type)
    parser = Stereo_Model.add_model_specific_args(parser)
    # add training data specific args
    parser = StereoDVS_woE_Loader.add_data_specific_args(parser)

    args = parser.parse_args()

    StereoDVS_Trainer_Default(args)

    main(args)
