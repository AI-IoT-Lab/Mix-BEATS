import torch
from torch import nn
from torch.nn import functional as F


class TSMixer(nn.Module):
    def __init__(self, patch_size, num_patches, patch_hidden_dim, time_hidden_dim):
        super(TSMixer, self).__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patch_mix = nn.Sequential(
            nn.LayerNorm(self.num_patches),
            nn.Linear(self.num_patches, patch_hidden_dim),
            nn.GELU(),
            nn.Linear(patch_hidden_dim, self.num_patches)
        )
        
        self.time_mix = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, time_hidden_dim),
            nn.GELU(),
            nn.Linear(time_hidden_dim, self.patch_size)
        )

        
        
    def forward(self, x):
        batch_size, context_length = x.shape
        x = x.view(batch_size, self.num_patches, self.patch_size)

        
        x = x + self.patch_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.time_mix(x)
        
        x = x.reshape(batch_size, self.num_patches * self.patch_size)
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim, thetas_dim, device, backcast_length=10, forecast_length=5, patch_size=8, num_patches=21):
        super(Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.device = device
        
        self.TSMixer = TSMixer(self.patch_size, self.num_patches, self.hidden_dim, self.hidden_dim)
        
        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):

        x = self.TSMixer(x)


        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
    

class Model(nn.Module):
    def __init__(
            self,
            device=torch.device('cpu'),
            num_blocks_per_stack=3,
            forecast_length=24,
            backcast_length=128,
            patch_size=8,
            num_patches=21,
            thetas_dim=8,
            hidden_dim=256,
            share_weights_in_stack=False
    ):
        super(Model, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.num_blocks_per_stack = num_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.thetas_dim = thetas_dim
        self.device = device
        self.stack_type = ['generic', 'generic', 'generic']

        self.parameters = []

        self.stacks = [self.create_stack(type) for type in self.stack_type]
        self.parameters = nn.ParameterList(self.parameters)

        self.to(self.device)

    def create_stack(self, type):
        if type == 'generic':
            blocks = []
            for _ in range(self.num_blocks_per_stack):
                block = Block(
                    self.hidden_dim, self.thetas_dim,
                    self.device, self.backcast_length, self.forecast_length, 
                    self.patch_size, self.num_patches,
                )
                self.parameters.extend(block.parameters())
                blocks.append(block)
            return blocks
        

    def forward(self, backcast):
        forecast = torch.zeros(backcast.size(0), self.forecast_length).to(self.device)
        for stack in self.stacks:
            for block in stack:
                backcast_block, forecast_block = block(backcast)
                backcast = backcast - backcast_block  
                forecast += forecast_block  
        return backcast, forecast


