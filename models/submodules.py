import torch
from networks.firenet import *
from networks.dispnet import *
from networks.dispnet_cor import *
from networks.adaptornet import *


def load_recon_model(model_type, path_to_model):
    if model_type == 'firenet':
        return load_firenet_model(path_to_model)


def load_firenet_model(path_to_model):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model)
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])

    return model


def load_stereo_model(model_type, path_to_model):
    if model_type == 'dispnet_v1':
        model = Dispnet_v1()
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {Dispnet_v1.__name__}')
        return model
    elif model_type == 'dispnet_v2':
        model = Dispnet_v2()
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {Dispnet_v2.__name__}')
        return model
    elif model_type == 'dispnet_4to2':
        model = Dispnet_4to2()
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {Dispnet_4to2.__name__}')
        return model
    elif model_type == 'dispnet_4to1':
        model = Dispnet_4to1()
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {Dispnet_4to1.__name__}')
        return model
    elif model_type == 'dispnet_2to2':
        model = Dispnet_2to2()
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {Dispnet_2to2.__name__}')
        return model
    # ===========================
    elif model_type == 'dispnet_cor_2to2':
        model = DispnetCorDual(1)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {DispnetCorDual.__name__}')
        return model
    elif model_type == 'dispnet_cor_4to2':
        model = DispnetCorDual(2)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {DispnetCorDual.__name__}')
        return model
    elif model_type == 'dispnet_cor_2to1':
        model = DispnetCor(1)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {DispnetCorDual.__name__}')
        return model
    elif model_type == 'dispnet_cor_4to1':
        model = DispnetCor(2)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        print(f'Successfully load {path_to_model} for model {DispnetCorDual.__name__}')
        return model


def load_adaptor_cls(model_type):
    if model_type == 'one2one_resnet':
        return AdaptorBaselineOne2One
    elif model_type == 'one2one_wt_left_resnet':
        return AdaptorBaselineOne2OneWtLeft
    elif model_type == 'one2one_wt_left_resnetv2':
        return AdaptorBaselineOne2OneWtLeftV2




