import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as tF
from argparse import ArgumentParser
from PIL import Image
import itertools

import pytorch_lightning as pl
from torchvision.utils import make_grid

from models.submodules import *
from networks.common import warping_disparity
from models.losses import edge_ware_smoothing, total_variation_sparse, total_variation, SSIM, GradSSIM, avg_endpoint_error, SSIM_L1, LPIPS_L1_Loss, gradient
from models.losses import RMSE, RMSE_log, Abs_Rel, Sq_Rel, AEPE, BadPixels, PSNR


class EventStereoSelfSuperviseGeneral(pl.LightningModule):
    """
    Baseline Model for Self-Supervised Event-based Stereo Matching training, only un-supervised trained on right event data and left intensity data
    Including:
    1. training code
    2. validation code
    3. testing code
    """
    def __init__(self,
                 optimizer='adam',
                 lr=0.001,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 scheduler='step',
                 scheduler_step_epoch=500,
                 scheduler_step_gamma=0.5,
                 scheduler_cos_T_max=20,
                 scheduler_cos_eta_min=0.000001,
                 stereo_model_type='general_recon_intensity',
                 stereo_model_path='pretrained_models/dispnetv2.pth',
                 smooth_loss='l1',
                 disparity_smoothing=1.,
                 grad_warping_consist_loss=1.,
                 disparity_consistency_lambda=0.3,
                 disparity_internal_lambda=0.3,
                 visualize_validation=True,
                 exp_name='NoName'):
        super(EventStereoSelfSuperviseGeneral, self).__init__()
        self.save_hyperparameters()
        self.init_stereo_network(stereo_model_type, stereo_model_path)
        self.init_loss()

        print('Loading Self supervised loss')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # ================= Optimizer
        parser.add_argument('--optimizer', type=str, default='adam', help='Used optimizer, [`adam`, `sgd`]')
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        # ================= Optimizer, Learning rate scheduler
        parser.add_argument('--scheduler', type=str, default='step', help='Learning rate scheduler, [`no`, `step`, `exp`, `linear`, `cos`]')
        parser.add_argument('--scheduler_step_epoch', type=int, default=30)
        parser.add_argument('--scheduler_step_gamma', type=float, default=0.5)
        parser.add_argument('--scheduler_cos_T_max', type=int, default=20)
        parser.add_argument('--scheduler_cos_eta_min', type=float, default=0.000001)
        # ================= Pretrained Components
        parser.add_argument('--stereo_model_type', type=str, default='dispnet_cor_2to1')
        parser.add_argument('--stereo_model_path', type=str, default='pretrained_models/dispnet_cor_2to1.pth')
        # ================= Training
        # Disparity Smoothness Loss
        parser.add_argument('--smooth_loss', type=str, default='EdgeWare', help='Loss function, [`l1`, `l2`, `EdgeWare`]')
        parser.add_argument('--disparity_smoothing', type=float, default=1.0)           # lambda
        # Gradient Warping Consistency Loss
        parser.add_argument('--grad_warping_consist_loss', type=float, default=1.0)     # lambda
        # The Disparity of Symmetrical information should be same
        parser.add_argument('--disparity_consistency_lambda', type=float, default=0.1)  # lambda
        # The Disparity between the same side should be zero
        parser.add_argument('--disparity_internal_lambda', type=float, default=0.1)     # lambda
        # Testing Tricks
        parser.add_argument('--visualize_validation', action='store_false', help='proportion of the RGB permuted pairs')
        # Model saving
        parser.add_argument('--exp_name', type=str, default='GSSL-Test;FireNet;DispNet:2to1;Loss:EdgeWareSmt(0.3)+Grad-Cons(1.0)-DispSym(0.3)+InternalDisp(0.3)', help='Experiment name')
        return parser

    @staticmethod
    def from_namespace(args):
        instance = EventStereoSelfSuperviseGeneral(
            optimizer=args.optimizer,
            lr=args.lr,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            scheduler=args.scheduler,
            scheduler_step_epoch=args.scheduler_step_epoch,
            scheduler_step_gamma=args.scheduler_step_gamma,
            scheduler_cos_T_max=args.scheduler_cos_T_max,
            scheduler_cos_eta_min=args.scheduler_cos_eta_min,
            stereo_model_type=args.stereo_model_type,
            stereo_model_path=args.stereo_model_path,
            smooth_loss=args.smooth_loss,
            disparity_smoothing=args.disparity_smoothing,
            grad_warping_consist_loss=args.grad_warping_consist_loss,
            disparity_consistency_lambda=args.disparity_consistency_lambda,
            disparity_internal_lambda=args.disparity_internal_lambda,
            visualize_validation=args.visualize_validation,
            exp_name=args.exp_name
        )
        return instance

    def init_loss(self):
        if self.hparams.smooth_loss == 'l1':
            self.smoothing_loss = total_variation_sparse
        elif self.hparams.smooth_loss == 'l2':
            self.smoothing_loss = total_variation
        elif self.hparams.smooth_loss == 'EdgeWare':
            self.smoothing_loss = edge_ware_smoothing
        else:
            self.smoothing_loss = total_variation

        self.consist_grad_loss = nn.L1Loss()
        self.disparity_loss = nn.L1Loss()
        self.ssim = SSIM()

    def init_stereo_network(self, stereo_model_type, stereo_model_path):
        self.stereo = load_stereo_model(stereo_model_type, stereo_model_path)

    def configure_scheduler(self, optimizer):
        if self.hparams.scheduler == 'no':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: self.hparams.lr)
        elif self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step_epoch, gamma=self.hparams.scheduler_step_gamma)
        elif self.hparams.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.scheduler_cos_T_max, eta_min=self.hparams.scheduler_cos_eta_min)
        else:
            print(f'Wrong scheduler parameter {self.hparams.scheduler}, using no scheduler')
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: self.hparams.lr)
        return scheduler

    def configure_optimizers(self):
        params = self.stereo.parameters()

        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr)
        else:
            print(f'Wrong optimizer parameter {self.hparams.optimizer}, using Adam')
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))
        scheduler = self.configure_scheduler(optimizer)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        # Get data from the dataset
        left_intensity, _right_intensity, _left_disparity, _right_disparity, right_reconstruction = batch

        # Calculate the disparity map for both left and right views
        pred_disparity_left, pred_disparity_right = self.calculate_disparity(left_intensity, right_reconstruction)

        # Now, gather both the recon version and intensity version for both left and right views,
        # _warped means this is obtained using warped from the other side
        # Right : right_reconstruction, right_intensity_warped
        # Left  : left_reconstruction_warped, left_intensity
        left_reconstruction_warped = warping_disparity(right_reconstruction, -1 * pred_disparity_left)
        right_intensity_warped = warping_disparity(left_intensity, pred_disparity_left)

        # Loss 1: Stereo(intensity_same_view, reconstruction_same_view) = 0
        pred_disparity_right_view_internal_1, pred_disparity_right_view_internal_2 = self.calculate_disparity(right_intensity_warped, right_reconstruction)
        pred_disparity_left_view_internal_1, pred_disparity_left_view_internal_2 = self.calculate_disparity(left_intensity, left_reconstruction_warped)
        loss_disparity_views_internal = (self.disparity_loss(pred_disparity_right_view_internal_1, torch.zeros_like(pred_disparity_right_view_internal_1)) +
                                        self.disparity_loss(pred_disparity_right_view_internal_2, torch.zeros_like(pred_disparity_right_view_internal_2)) +
                                        self.disparity_loss(pred_disparity_left_view_internal_1, torch.zeros_like(pred_disparity_left_view_internal_1)) +
                                        self.disparity_loss(pred_disparity_left_view_internal_2, torch.zeros_like(pred_disparity_left_view_internal_2))) / 4

        # Loss 2: Stereo(left_view, right_view) is same for both way to calculate
        pred_disparity_left_Warped_left, pred_disparity_left_Warped_right = self.calculate_disparity(right_intensity_warped, left_reconstruction_warped)
        loss_disparity_consistency = (self.disparity_loss(pred_disparity_left_Warped_right, pred_disparity_left) +
                                     self.disparity_loss(pred_disparity_left_Warped_left, pred_disparity_right)) / 2

        # Loss 3: Smoothing Loss
        disparity_tv_loss_left = self.smoothing_loss(pred_disparity_left) if self.hparams.smooth_loss != 'EdgeWare' else self.smoothing_loss(pred_disparity_right, left_intensity)
        disparity_tv_loss_right = self.smoothing_loss(pred_disparity_right) if self.hparams.smooth_loss != 'EdgeWare' else self.smoothing_loss(pred_disparity_right, right_reconstruction)
        disparity_tv_loss = (disparity_tv_loss_left + disparity_tv_loss_right) / 2

        # Gradient Consistency Loss to Avoid trivial solutions
        right_reconstruction_gradient = gradient(right_reconstruction)
        left_intensity_gradient = gradient(left_intensity)
        left_reconstruction_warped_gradient = warping_disparity(right_reconstruction_gradient, -1 * pred_disparity_left)
        right_intensity_warped_gradient = warping_disparity(left_intensity_gradient, pred_disparity_left)

        grad_warping_consist_loss = 1 - (self.ssim(right_reconstruction_gradient, right_intensity_warped_gradient) +
                                         self.ssim(left_reconstruction_warped_gradient, left_intensity_gradient)) / 2

        loss = disparity_tv_loss * self.hparams.disparity_smoothing + \
               loss_disparity_consistency * self.hparams.disparity_consistency_lambda + \
               loss_disparity_views_internal * self.hparams.disparity_internal_lambda + \
               grad_warping_consist_loss * self.hparams.grad_warping_consist_loss

        self.logger.experiment.add_scalar('train_loss', loss, self.global_step)
        self.logger.experiment.add_scalar('gradient_consistency_loss', grad_warping_consist_loss, self.global_step)
        self.logger.experiment.add_scalar('disparity_tv_loss', disparity_tv_loss, self.global_step)
        self.logger.experiment.add_scalar('loss_disparity_consistency', loss_disparity_consistency, self.global_step)
        self.logger.experiment.add_scalar('loss_disparity_views_internal', loss_disparity_views_internal, self.global_step)

        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = dict(self.hparams)
        checkpoint['global_step'] = self.global_step
        checkpoint['global_epoch'] = self.current_epoch
        checkpoint['stereo'] = self.stereo.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.stereo.load_state_dict(checkpoint['stereo'])
        self.hparams.update(checkpoint['hparams'])

    def calculate_disparity(self, left_intensity, right_reconstruction):
        left_intensity_r = torch.flip(left_intensity, dims=[3])
        right_reconstruction_r = torch.flip(right_reconstruction, dims=[3])
        # TODO: need to be changed if it fails
        if self.hparams.stereo_model_type in ['general_recon_intensity', 'dispnet_cor_2to1']:
            pred_disparity_left = self.stereo(left_intensity, right_reconstruction)
            pred_disparity_right_r = self.stereo(right_reconstruction_r, left_intensity_r)
            pred_disparity_right = torch.flip(pred_disparity_right_r, dims=[3])
        else:
            raise Exception('Check the dispnet')
        return pred_disparity_left, pred_disparity_right

    def validation_step(self, batch, batch_idx):
        left_intensity, _right_intensity, _left_disparity, _right_disparity, right_reconstruction = batch
        batch_size = _right_intensity.shape[0]

        error_mask = torch.where(torch.logical_or(torch.isnan(_left_disparity), torch.isinf(_left_disparity)), 0., 1.)
        _right_disparity = torch.where(torch.isnan(_right_disparity), torch.full_like(_right_disparity, 0), _right_disparity)
        _right_disparity = torch.where(torch.isinf(_right_disparity), torch.full_like(_right_disparity, 0), _right_disparity)
        _left_disparity = torch.where(torch.isnan(_left_disparity), torch.full_like(_left_disparity, 0), _left_disparity)
        _left_disparity = torch.where(torch.isinf(_left_disparity), torch.full_like(_left_disparity, 0), _left_disparity)

        # Calculate the disparity map for both left and right views
        pred_disparity_left, pred_disparity_right = self.calculate_disparity(left_intensity, right_reconstruction)

        # Now, gather both the recon version and intensity version for both left and right views,
        # _warped means this is obtained using warped from the other side
        # Right : right_reconstruction, right_intensity_warped
        # Left  : left_reconstruction_warped, left_intensity
        left_reconstruction_warped = warping_disparity(right_reconstruction, -1 * pred_disparity_left)
        right_intensity_warped = warping_disparity(left_intensity, pred_disparity_left)

        # Loss 1: Stereo(intensity_same_view, reconstruction_same_view) = 0
        pred_disparity_right_view_internal_1, pred_disparity_right_view_internal_2 = self.calculate_disparity(right_intensity_warped, right_reconstruction)
        pred_disparity_left_view_internal_1, pred_disparity_left_view_internal_2 = self.calculate_disparity(left_intensity, left_reconstruction_warped)

        # Loss 2: Stereo(left_view, right_view) is same for both way to calculate
        pred_disparity_left_Warped_left, pred_disparity_left_Warped_right = self.calculate_disparity(right_intensity_warped, left_reconstruction_warped)

        # Gradient Consistency Loss to Avoid trivial solutions
        right_reconstruction_gradient = gradient(right_reconstruction)
        right_intensity_warped_gradient = gradient(right_intensity_warped)
        left_reconstruction_warped_gradient = gradient(left_reconstruction_warped)
        left_intensity_gradient = gradient(left_intensity)

        quantity_list = []

        aepe = AEPE(pred_disparity_left, _left_disparity, error_mask)
        d_1, d_3, d_5 = BadPixels(pred_disparity_left, _left_disparity, error_mask)
        quantity_list.append(aepe)
        quantity_list.append(d_1)
        quantity_list.append(d_3)
        quantity_list.append(d_5)

        if self.hparams.visualize_validation:
            if batch_idx == 0:
                for i in range(left_intensity.shape[0]):
                    disp_vis_norm_max = max(pred_disparity_left[i: i+1, :, :, :].max(),
                                            pred_disparity_right[i: i+1, :, :, :].max(),
                                            pred_disparity_left_Warped_left[i: i+1, :, :, :].max(),
                                            pred_disparity_left_Warped_right[i: i+1, :, :, :].max(),
                                            pred_disparity_right_view_internal_1[i: i+1, :, :, :].max(),
                                            pred_disparity_right_view_internal_2[i: i+1, :, :, :].max(),
                                            pred_disparity_left_view_internal_1[i: i+1, :, :, :].max(),
                                            pred_disparity_left_view_internal_2[i: i+1, :, :, :].max(),
                                            _left_disparity[i: i+1, :, :, :].max(),
                                            _right_intensity[i: i+1, :, :, :].max())

                    image_stake_tensor = torch.cat([
                        left_intensity[i: i+1, :, :, :],
                        right_intensity_warped[i: i+1, :, :, :],
                        torch.clamp(pred_disparity_left[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                        torch.clamp(pred_disparity_right[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,

                        left_reconstruction_warped[i: i+1, :, :, :],
                        right_reconstruction[i: i+1, :, :, :],
                        torch.clamp(pred_disparity_left_Warped_left[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                        torch.clamp(pred_disparity_left_Warped_right[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,

                        left_reconstruction_warped_gradient[i: i+1, :, :, :],
                        right_intensity_warped_gradient[i: i+1, :, :, :],
                        torch.clamp(_left_disparity[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                        torch.clamp(_right_intensity[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,

                        torch.clamp(pred_disparity_right_view_internal_1[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                        torch.clamp(pred_disparity_right_view_internal_2[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                        torch.clamp(pred_disparity_left_view_internal_1[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                        torch.clamp(pred_disparity_left_view_internal_2[i: i+1, :, :, :], min=0.) / disp_vis_norm_max,
                    ], dim=0)
                    self.logger.experiment.add_image(f'image_{batch_size + i}_valid', make_grid(image_stake_tensor, nrow=4, padding=0), self.global_step)

        return quantity_list

    def validation_epoch_end(self, val_step_outputs):
        mean_aepe, mean_d_1, mean_d_3, mean_d_5 = torch.mean(torch.Tensor(val_step_outputs), dim=0)
        self.logger.experiment.add_scalar(f'validation_avg_AEPE', mean_aepe, self.global_step)
        self.logger.experiment.add_scalar(f'validation_avg_D_1', mean_d_1, self.global_step)
        self.logger.experiment.add_scalar(f'validation_avg_D_3', mean_d_3, self.global_step)
        self.logger.experiment.add_scalar(f'validation_avg_D_5', mean_d_5, self.global_step)


    def test_step(self, batch, batch_idx):

        quantity_list = self.validation_step(batch, batch_idx) # aepe, d1, d_3, d_5
        return quantity_list
            

    def test_epoch_end(self, test_step_output):
        mean_aepe, mean_d_1, mean_d_3, mean_d_5 = torch.mean(torch.Tensor(test_step_output), dim=0)

        print(f'Test_avg EPE', mean_aepe, self.global_step)
        print(f'Test_avg_D_1', mean_d_1, self.global_step)
        print(f'Test_avg_D_3', mean_d_3, self.global_step)
        print(f'Test_avg_D_5', mean_d_5, self.global_step)
