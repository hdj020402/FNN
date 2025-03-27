import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict
from nets.FNN import FNN

def gen_model(param: Dict, dataset, ) -> FNN:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_layers = param['hidden_layer']
    if param['default_feature']['ECFP']['enabled'] is True:
        num_input = len(param['feature_list']) + param['default_feature']['ECFP']['nBits']
    else:
        num_input = len(param['feature_list'])
    num_output = len(param['target_list'])
    dims = [num_input, *hidden_layers, num_output]
    net = FNN(dims)
    model = net.to(device)
    return model

def gen_optimizer(param: Dict, model: FNN) -> torch.optim.Optimizer:
    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr = param['lr'])
    return optimizer

def gen_scheduler(param: Dict, optimizer) -> ReduceLROnPlateau | None:
    if param['scheduler']['type'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode = 'min',
            factor = param['scheduler']['factor'], patience = param['scheduler']['patience'],
            min_lr = param['scheduler']['min_lr']
            )
    else:
        print('No scheduler')
        scheduler = None
    return scheduler
