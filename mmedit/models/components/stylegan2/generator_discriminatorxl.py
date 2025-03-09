# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix
from mmedit.models.registry import COMPONENTS
# from torch_utils import misc
# import dnnlib
# import legacy
from mmedit.models.registry import BACKBONES
 
@COMPONENTS.register_module()
class SynthesisNetworkv2(torch.nn.Module):
    def __init__(self,pretrained_stylegan='/output/v-huiminzeng/imagenet256.pkl'):
        super().__init__()    

        # net_dict=torch.load('/output/v-huiminzeng/stylegan_xl/sythesis_backbone.pth', map_location='cpu')
        # net, net_state_dict = net_dict["net"], net_dict["state_dict"]


        with dnnlib.util.open_url(pretrained_stylegan) as f:
        # with dnnlib.util.open_url('/home/v-huiminzeng/stylegan-xl/pretrained_models/imagenet256.pkl') as f:
            net = legacy.load_network_pkl(f)['G_ema'].synthesis
            net = net.eval().requires_grad_(False) 
            f.close()

  
        self.num_ws=net.num_ws
        self.w_dim=net.w_dim
        self.img_channels=net.img_channels
        self.img_resolution=net.img_resolution
        self.output_scale=net.output_scale
        self.layer_names=net.layer_names
        self.input=net.input
        for layer in net.layer_names:
            setattr(self, layer, getattr(net,layer))
 
        # self.load_state_dict(net_state_dict, strict=True)
        self.load_state_dict(net.state_dict(), strict=True)
 
 
    def forward(self, ws, **layer_kwargs):
        # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.view(ws.size(0), -1, self.w_dim)
        ws = ws.to(torch.float32).unbind(dim=1)
        
        # Execute layers.
        x = self.input(ws[0])

        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.to(torch.float32)
        return x
    
 
 
@COMPONENTS.register_module()
class SynthesisNetworkv3(SynthesisNetworkv2):
 
    def forward(self, ws, fea_list_name, **layer_kwargs):
        feat_list=[]
        # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.view(ws.size(0), -1, self.w_dim)
        ws = ws.to(torch.float32).unbind(dim=1)
        
        # Execute layers.
        x = self.input(ws[0])

        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
            if str(name) in fea_list_name:
                feat_list.append(x.to(torch.float32))
            if str(name) == fea_list_name[-1]:
                break
        assert len(feat_list)==len(fea_list_name)
        return feat_list
        

