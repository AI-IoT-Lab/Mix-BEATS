import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer






class GenericBlock(nn.Module):
    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(GenericBlock, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.device = device
        
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        
        self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
    


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor





class Model(nn.Module):
    GENERIC_BLOCK = 'generic'

    def __init__(
            self,
            configs
    ):
        super(Model, self).__init__()
        self.forecast_length = configs['pred_len']
        self.backcast_length = configs['seq_len']
        self.hidden_layer_units = configs['hidden_dim']
        self.num_blocks_per_stack = configs['num_blocks_per_stack']
        self.stacks = []
        self.thetas_dim = configs['thetas_dim']
        self.parameters = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for stack_id in range(3):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self.task_name = configs['task_name']

    def create_stack(self, stack_id):
        blocks = []
        for _ in range(self.num_blocks_per_stack):
            block = GenericBlock(
                    self.hidden_layer_units, self.thetas_dim,
                    self.device, self.backcast_length, self.forecast_length
                )
            self.parameters.extend(block.parameters())
            blocks.append(block)
        return blocks




    def short_forecast(self, backcast):
        backcast = squeeze_last_dim(backcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,)) 
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        return backcast, forecast
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'short_term_forecast':
            backcast = x_enc.squeeze(-1)
            backcast, forecast = self.short_forecast(backcast)
            forecast = forecast.unsqueeze(-1)
            return forecast
        return None




