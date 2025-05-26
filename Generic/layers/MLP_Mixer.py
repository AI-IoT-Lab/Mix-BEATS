import torch
from torch import nn

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