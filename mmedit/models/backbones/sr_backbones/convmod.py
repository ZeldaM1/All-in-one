import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.norm(x)   
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x
import matplotlib.pyplot as plt
import numpy as np
class Ghost_Spatial_Attn(nn.Module):
    def __init__(self, dim, sigmoid=True,softmax=False):
        super().__init__()
 
        self.a = nn.Sequential( 
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=(1,5), stride=1, padding=(0,2), groups=dim, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=(5,1), stride=1, padding=(2,0), groups=dim, bias=False),
            ) 
        if sigmoid:
            self.sigmoid=nn.Sigmoid()
 
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.idx=0

    def forward(self, x): 
        # a = self.a[:3](x)
        
        
        a = self.a(x)
        # breakpoint()
        # save_tensor=a.cpu().numpy()[0].mean(axis=0)
        # norm=(save_tensor-np.min(save_tensor))/np.max(save_tensor)-np.min(save_tensor)
        # plt.imsave(f'./visual_ablation_s/attn_{self.idx}.png',norm,cmap='viridis')

        v = self.v(x)
        if hasattr(self,'sigmoid'):
            a=self.sigmoid(a)
        
        a = F.interpolate(a, size=(v.shape[-2],v.shape[-1]),mode='nearest') 
        return self.proj(a*v)

class Ghost_Spatial_Attnv3(nn.Module):
    def __init__(self, dim, sigmoid=True,kernel_size=5,softmax=False):
        super().__init__()
 
        self.a = nn.Sequential(
                    nn.Conv2d(dim, dim, 1),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size, padding=2, groups=dim, bias=False)
            )
        
        if sigmoid:
            self.sigmoid=nn.Sigmoid()
 
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.idx=0

    def forward(self, x): 
 
        a = self.a(x)
 
        v = self.v(x)
        if hasattr(self,'sigmoid'):
            a=self.sigmoid(a)
        
        a = F.interpolate(a, size=(v.shape[-2],v.shape[-1]),mode='nearest') 
        return self.proj(a*v)

class Ghost_Spatial_Attnv4(nn.Module):
    def __init__(self, dim, sigmoid=True,kernel_size=5,softmax=False):
        super().__init__()
 
        self.a = nn.Sequential(
                    nn.Conv2d(dim, dim, 1),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size, padding=2, bias=False)
            )
        
        if sigmoid:
            self.sigmoid=nn.Sigmoid()
 
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.idx=0

    def forward(self, x): 
 
        a = self.a(x)
 
        v = self.v(x)
        if hasattr(self,'sigmoid'):
            a=self.sigmoid(a)
        
        a = F.interpolate(a, size=(v.shape[-2],v.shape[-1]),mode='nearest') 
        return self.proj(a*v)

 
    
class ConvMod_Spatial_Attn(nn.Module):
    def __init__(self, dim, kernel_size=11):
        super().__init__()
        if kernel_size==11:
            self.a = nn.Sequential(
                    nn.Conv2d(dim, dim, 1),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size, padding=5, groups=dim)
            )
        elif kernel_size==5:
            self.a = nn.Sequential(
                    nn.Conv2d(dim, dim, 1),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size, padding=2, groups=dim)
            )
        self.v = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x): 
        a = self.a(x)
        # breakpoint()
        # plt.imsave(f'./visual_ablation_s/attn_{self.idx}.png',a.cpu().numpy()[0].mean(axis=0),cmap='turbo')
        x = a * self.v(x)
 
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        
        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
      
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
