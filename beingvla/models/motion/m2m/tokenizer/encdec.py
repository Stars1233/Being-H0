import torch
import torch.nn as nn
from typing import Optional, Literal


class Swish(nn.Module):
    """Swish activation function (x * sigmoid(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """Residual 1D convolutional block with optional normalization."""
    NORM_MAP = {
        "LN": nn.LayerNorm,
        "GN": lambda n_in: nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True),
        "BN": lambda n_in: nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True),
    }
    
    ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "silu": Swish,
        "gelu": nn.GELU
    }
    
    def __init__(self, 
                 n_in: int, 
                 n_state: int, 
                 dilation: int = 1, 
                 activation: Literal['relu', 'silu', 'gelu'] = 'silu',
                 norm: Optional[Literal['LN', 'GN', 'BN']] = None):
        super().__init__()
  
        self.norm = norm
        self.norm1 = self._create_norm_layer(norm, n_in)
        self.norm2 = self._create_norm_layer(norm, n_in)

        padding = dilation
 
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)

        self.activation1 = self.ACTIVATION_MAP[activation]()
        self.activation2 = self.ACTIVATION_MAP[activation]()
      
    def _create_norm_layer(self, norm, n_in):
        return self.NORM_MAP.get(norm, nn.Identity)(n_in)
 
    def forward(self, x):
        x_orig = x

        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)  
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        x = self.conv2(x)

        return x + x_orig


class Resnet1D(nn.Module):
    """1D ResNet with configurable dilation rates."""
    def __init__(self, 
                 n_in: int, 
                 n_depth: int, 
                 dilation_growth_rate: int = 1, 
                 reverse_dilation: bool = True,
                 activation: str = 'silu',
                 norm: Optional[str] = None):
        super().__init__()
        
        blocks = [
            ResConv1DBlock(
                n_in, n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm
            ) for depth in range(n_depth)
        ]
        
        self.model = nn.Sequential(*(blocks[::-1] if reverse_dilation else blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """1D convolutional encoder with downsampling."""
    
    def __init__(self,
                 input_emb_width: int = 3,
                 output_emb_width: int = 512,
                 down_t: int = 3,
                 stride_t: int = 2,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 norm: Optional[str] = None,
                 num_conv_layers = 1):
        super().__init__()
        
        
        filter_t, pad_t = stride_t * 2, stride_t // 2

        blocks = []
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(num_conv_layers-1):
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 num_conv_layers=1):
        super().__init__()

        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(num_conv_layers-1):
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())

        for _ in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    

