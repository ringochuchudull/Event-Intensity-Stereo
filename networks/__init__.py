from networks.conv3d import CMNet3DConvD16S8


NETWORK_NAME_DICT = {
    'CMNet3DConvD16S8': CMNet3DConvD16S8
}


def print_network(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    num_params = num_params / 1000
    print(f'Network {model.__class__.__name__} was created. Total number of parameters: {num_params:.1f} kelo. '
          'To see the architecture, do print(network).')


def load_network(network_name):
    if network_name in NETWORK_NAME_DICT.keys():
        net_cls = NETWORK_NAME_DICT[network_name]
    else:
        raise Exception(f'Check your network name, {network_name} is not in the following available networks: \n{NETWORK_NAME_DICT.keys()}')
    return net_cls
