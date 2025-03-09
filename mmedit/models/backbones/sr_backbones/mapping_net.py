from enum import Enum
import math
import numpy as np
import torch
from torch import nn
# from torch_utils import misc
# import dnnlib
# import legacy

from .dmci_styleganxl_cascade_v2 import Mapping_control
# from mmedit.models.backbones.sr_backbones.dmci_styleganxl_cascade_v2 import Mapping_control
 

class Mapping_control_z(nn.Module):
    def __init__(self,  out_res=8, out_dim=512, inplace=True, **kwargs):
        # # re_list=[8 16 32 64 128 256]
        super().__init__()
        self.out_res=out_res
        self.down_feas=Mapping_control(out_res=out_res, out_dim=out_dim, inplace=inplace, **kwargs)
 
        self.to_z0 = nn.Sequential( #  256 8
            nn.Conv2d(256, 256, 3, 2, 1, bias=True),nn.LeakyReLU(inplace=inplace),      #  256 4
            nn.Conv2d(256, 256, 3, 2, 1, bias=True),nn.LeakyReLU(inplace=inplace),      # 256 2
            nn.AdaptiveAvgPool2d((1,1))# 256 x 1 x 1
            )   # 64 x 2 x 2
        
        self.to_z1 =  nn.Linear(256, 128)  # 1x 256  -> 1 x 64
  
    def forward(self, latent):
        latent,control_list=self.down_feas(latent)
        latent=latent.view(latent.shape[0],-1,self.out_res,self.out_res)
        z0= self.to_z0(latent)
        z_noise=self.to_z1(z0.view(-1,256))
 
        return z_noise,control_list

 
# class Mapping_control_vec(nn.Module):
#     def __init__(self, input_res=256, tar_dim=128, base_dim=16, max_dim=512, inplace=True, **kwargs):
#         # # re_list=[8 16 32 64 128 256]
#         super().__init__()
#         self.input_layer=nn.Sequential(nn.Conv2d(3, base_dim, 3, 1, 1, bias=True),nn.LeakyReLU(inplace=inplace))
#         self.log_size = int(np.log2(input_res))
#         pre_ch=base_dim
#         for i in range(1,self.log_size+1):
#             input_res=input_res//2
#             cur_ch=min(base_dim*i,max_dim)
#             layer=nn.Sequential(nn.Conv2d(pre_ch, cur_ch, 3, 2, 1, bias=True),nn.AdaptiveAvgPool2d(input_res), nn.LeakyReLU(inplace=inplace))
#             setattr(self,f"down_layer{i}",layer)
#             pre_ch=cur_ch
        
#         self.to_z0 = nn.AdaptiveAvgPool2d(1)
#         # flatten  input.view(input.size(0), -1)
#         self.to_z1 = nn.Linear(max_dim, tar_dim)

#     def forward(self, img):
#         latent = self.input_layer(img)
#         for i in range(1,self.log_size+1):
#             latent=getattr(self,f"down_layer{i}")(latent)
#         z0= self.to_z0(latent)
#         z0_flatten=z0.view(z0.size(0), -1)
#         vec=self.to_z1(z0_flatten)
#         return vec

