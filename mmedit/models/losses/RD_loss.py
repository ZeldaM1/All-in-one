# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import masked_loss

 
def get_loss_func(loss_func):
    # pylint: disable=possibly-unused-variable

    def get_final_loss(costs, rate_number):
        losses = costs
        loss = torch.sum(losses) / rate_number
        return {
            'costs': costs,
            'losses': losses,
            'loss': loss,
        }

    def loss_me_mse(rd, lmbdas):
        costs = lmbdas * rd['me_mse']
        return get_final_loss(costs, len(lmbdas))

    def loss_me_rdc_mse(rd, lmbdas):
        costs = lmbdas * rd['me_mse'] + rd['bpp_mv_y'] + rd['bpp_mv_z']
        return get_final_loss(costs, len(lmbdas))

    def loss_recon_mse(rd, lmbdas):
        costs = lmbdas * rd['mse']
        return get_final_loss(costs, len(lmbdas))

    def loss_recon_rdc_mse(rd, lmbdas):
        costs = lmbdas * rd['mse'] + rd['bpp_y'] + rd['bpp_z']
        return get_final_loss(costs, len(lmbdas))

    def loss_total_rdc_mse(rd, lmbdas):
        costs = lmbdas * rd['mse'] + rd['bpp']
        return get_final_loss(costs, len(lmbdas))

    def loss_total_rdc_ms_ssim(rd, lmbdas):
        costs = lmbdas / 17 * (rd['1-ssim']) + rd['bpp']
        return get_final_loss(costs, len(lmbdas))

    loss_func_name = f'loss_{loss_func}'
    assert loss_func_name in locals()
    return locals()[loss_func_name]

 
@LOSSES.register_module()
class RDLoss(nn.Module):
    def __init__(self,loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, rd, lmbdas, loss_type):
        loss_func = get_loss_func(loss_type)
        loss_dict=loss_func(rd, lmbdas)
        return self.loss_weight * loss_dict["loss"], loss_dict
 