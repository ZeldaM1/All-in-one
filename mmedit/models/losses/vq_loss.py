# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pyiqa
from ..registry import LOSSES
from .utils import masked_loss
from mmedit.utils import ycbcr2rgb
_reduction_modes = ['none', 'mean', 'sum']
 
@LOSSES.register_module()
class LPIPSLoss(nn.Module):
    def __init__(self, loss_weight = 1.0, yuv_input=False):
        super(LPIPSLoss, self).__init__()
        self.model = pyiqa.create_metric('lpips-vgg', as_loss=True)
        self.loss_weight = loss_weight
        self.yuv_input=yuv_input
        

    def forward(self, x, gt):
        if self.yuv_input:
            x=ycbcr2rgb(x)
            gt=ycbcr2rgb(gt)

        return self.model(x, gt) * self.loss_weight, None


 
@LOSSES.register_module()
class Codebook_SemanticLoss(nn.Module):
    def __init__(self, loss_weight = 1.0):
        super(Codebook_SemanticLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, l_codebook):
        return l_codebook * self.loss_weight 

 