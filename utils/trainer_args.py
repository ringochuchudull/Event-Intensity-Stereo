from collections import OrderedDict

TRAINER_DEFAULTS = OrderedDict({
    'default_root_dir': None,
    'gradient_clip_val': 0,
    'process_position': 0,
    'num_nodes': 1,
    'num_processes': 1,
    'gpus': None,
    'auto_select_gpus': False,
    'tpu_cores': None,
    'log_gpu_memory': None,
    'progress_bar_refresh_rate': 1,
    'overfit_batches': 0.0,
    'track_grad_norm': -1,
    'check_val_every_n_epoch': 1,
    'fast_dev_run': False,
    'accumulate_grad_batches': 1,
    'max_epochs': 1000,
    'min_epochs': 1,
    'max_steps': None,
    'min_steps': None,
    'limit_train_batches': 1.0,
    'limit_val_batches': 1.0,
    'limit_test_batches': 1.0,
    'val_check_interval': 1.0,
    'flush_logs_every_n_steps': 100,
    'log_every_n_steps': 50,
    'accelerator': None,
    'sync_batchnorm': False,
    'precision': 32,
    'weights_summary': 'top',
    'weights_save_path': None,
    'num_sanity_val_steps': 2,
    'truncated_bptt_steps': None,
    'resume_from_checkpoint': None,
    'profiler': None,
    'benchmark': False,
    'deterministic': False,
    'reload_dataloaders_every_epoch': False,
    'auto_lr_find': False,
    'replace_sampler_ddp': True,
    'terminate_on_nan': False,
    'auto_scale_batch_size': False,
    'prepare_data_per_node': True,
    'plugins': None,
    'amp_backend': 'native',
    'amp_level': 'O2',
    'distributed_backend': None,
    # 'automatic_optimization': None,
    'move_metrics_to_cpu': False,
    # 'enable_pl_optimizer': False,
})


class Bunch(object):  #
    def __init__(self, adict):
        self.__dict__.update(adict)


CM_TRAINER_DEFAULTS = OrderedDict({
    'default_root_dir': './EXPs/CMNet3DConvD16S8',
    'gpus': '1',  # This is gpu usage
    'log_gpu_memory': True,
    'progress_bar_refresh_rate': 0,
    'overfit_batches': 0.0,
    'track_grad_norm': 2,
    'check_val_every_n_epoch': 1,  # check validation per this epochs
    'max_epochs': 2000,  # with 32 batch_size, DIV2K has 113 iters ber epoch
    'min_epochs': 1,
    'weights_summary': 'top',
    'weights_save_path': './EXPs/CMNet3DConvD16S8',
    'resume_from_checkpoint': None,
    'profiler': 'simple',  # check profiler when debugging
    'terminate_on_nan': True,
    'num_sanity_val_steps': -1,
    # 'auto_scale_batch_size': 'binsearch',  # run batch size scaling, result overrides hparams.batch_size
})


def cm_Trainer_Default(args):
    trainer_dict = vars(args)
    for key in TRAINER_DEFAULTS.keys():
        if trainer_dict[key] == TRAINER_DEFAULTS[key] and key in CM_TRAINER_DEFAULTS.keys():
            trainer_dict[key] = CM_TRAINER_DEFAULTS[key]
    return Bunch(trainer_dict)


StereoDVS_TRAINER_DEFAULTS = OrderedDict({
    'default_root_dir': './EXPs/Stereo_GeneralTest',
    'gpus': '1',  # This is gpu usage
    'log_gpu_memory': True,
    'progress_bar_refresh_rate': 0,
    'overfit_batches': 0.0,
    'track_grad_norm': -1,
    'check_val_every_n_epoch': 1,  # check validation per this epochs
    'accumulate_grad_batches': 1,
    'max_epochs': 2000,  # with 32 batch_size, DIV2K has 113 iters ber epoch
    'min_epochs': 1,
    'weights_summary': 'top',
    'resume_from_checkpoint': None,
    'profiler': 'simple',  # check profiler when debugging
    'terminate_on_nan': True,
    'num_sanity_val_steps': 1,
    # 'auto_scale_batch_size': 'binsearch',  # run batch size scaling, result overrides hparams.batch_size
})


def StereoDVS_Trainer_Default(args):
    trainer_dict = vars(args)
    for key in TRAINER_DEFAULTS.keys():
        if trainer_dict[key] == TRAINER_DEFAULTS[key] and key in StereoDVS_TRAINER_DEFAULTS.keys():
            trainer_dict[key] = StereoDVS_TRAINER_DEFAULTS[key]
    return Bunch(trainer_dict)
