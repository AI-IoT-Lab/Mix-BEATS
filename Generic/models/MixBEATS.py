import torch
from torch import nn
from torch.nn import functional as F
from layers.MLP_Mixer import TSMixer



class GenericBlock(nn.Module):
    def __init__(self, hidden_dim, thetas_dim, device, backcast_length=10, forecast_length=5, patch_size=8, num_patches=21):
        super(GenericBlock, self).__init__()
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
            configs
    ):
        super(Model, self).__init__()
        self.forecast_length = configs['pred_len']
        self.backcast_length = configs['seq_len']
        self.patch_size = configs['patch_size']
        self.num_patches = configs['seq_len'] // configs['patch_size']
        self.hidden_dim = configs['hidden_dim']
        self.num_blocks_per_stack = configs['num_blocks_per_stack']
        self.thetas_dim = configs['thetas_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stack_type = ['generic', 'generic', 'generic']
        self.task_name = configs['task_name']
        self.scaling = configs['scaling']

        self.parameters = []

        self.stacks = [self.create_stack(type) for type in self.stack_type]
        self.parameters = nn.ParameterList(self.parameters)

        self.to(self.device)
        self._loss = None
        self._opt = None

    def create_stack(self, type):
        if type == 'generic':
            blocks = []
            for i in range(self.num_blocks_per_stack):
                block = GenericBlock(
                    self.hidden_dim, self.thetas_dim,
                    self.device, self.backcast_length, self.forecast_length, 
                    self.patch_size, self.num_patches
                )
                self.parameters.extend(block.parameters())
                blocks.append(block)
            return blocks


    def short_forecast(self, backcast):
        forecast = torch.zeros(backcast.size(0), self.forecast_length).to(self.device)
        for stack in self.stacks:
            for block in stack:
                backcast_block, forecast_block = block(backcast)
                backcast = backcast - backcast_block  
                forecast += forecast_block  
        return backcast, forecast
    


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'short_term_forecast':
            backcast = x_enc.squeeze(-1)
            backcast, forecast = self.short_forecast(backcast)
            forecast = forecast.unsqueeze(-1)
            return forecast
        return None


