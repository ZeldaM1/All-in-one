# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import masked_loss

 
def get_loss_func(loss_func):
    # pylint: disable=possibly-unused-variable
  
    def loss_me_mse(rd):
        return rd['me_mse'],0
 
    def loss_me_rdc_mse(rd):
        return rd['me_mse'],(rd['bpp_mv_y'] + rd['bpp_mv_z'])
 

    def loss_recon_mse(rd):
        return rd['mse'],0
 
    def loss_recon_rdc_mse(rd):
        return rd['mse'],(rd['bpp_y'] + rd['bpp_z'])
 

    def loss_total_rdc_mse(rd):
        return rd['mse'], rd['bpp']
 

    def loss_total_rdc_ms_ssim(rd):
        return 17 * (rd['1-ssim']), rd['bpp']
 
    loss_func_name = f'loss_{loss_func}'
    assert loss_func_name in locals()
    return locals()[loss_func_name]

 
@LOSSES.register_module()
class LambdaDLoss(nn.Module):
    def __init__(self,loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, rd, loss_type):
        loss_func = get_loss_func(loss_type)
        distortion_loss, bpp_loss =loss_func(rd)
        return self.loss_weight * distortion_loss, bpp_loss
 