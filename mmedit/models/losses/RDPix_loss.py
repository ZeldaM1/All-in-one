# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from mmedit.utils import yuv_444_to_420
 
def get_mse(self, x, x_hat, yuv420):
    _, _, H, W = x.size()
    pixel_num = H * W
    if yuv420:
        org_y, org_u, org_v = yuv_444_to_420(x)
        rec_y, rec_u, rec_v = yuv_444_to_420(x_hat)
        mse_y = self.mse(reduction='none')(org_y, rec_y)
        mse_u = self.mse(reduction='none')(org_u, rec_u)
        mse_v = self.mse(reduction='none')(org_v, rec_v)
        mse_y = torch.sum(mse_y, dim=(1, 2, 3)) / pixel_num
        mse_u = torch.sum(mse_u, dim=(1, 2, 3)) / pixel_num * 4
        mse_v = torch.sum(mse_v, dim=(1, 2, 3)) / pixel_num * 4
        mse = (4 * mse_y + mse_u + mse_v) / 6 * 3   # rgb is sum, not average MSE
    else:
        mse = self.mse(x, x_hat)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
    return mse
  
 
@LOSSES.register_module()
class Rate_PixLoss(nn.Module):
    def __init__(self,loss_weight: float = 1.0,yuv420=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse=nn.MSELoss(reduction='none')
        self.yuv420=yuv420
         
    def forward(self, pred, target, lambda_w=None, **kwargs):
        _, _, H, W = pred.size()
        pixel_num = H * W
        if self.yuv420:
            org_y, org_u, org_v = yuv_444_to_420(pred)
            rec_y, rec_u, rec_v = yuv_444_to_420(target)
            mse_y = self.mse(org_y, rec_y)
            mse_u = self.mse(org_u, rec_u)
            mse_v = self.mse(org_v, rec_v)
            mse_y = torch.sum(mse_y, dim=(1, 2, 3)) / pixel_num
            mse_u = torch.sum(mse_u, dim=(1, 2, 3)) / pixel_num * 4
            mse_v = torch.sum(mse_v, dim=(1, 2, 3)) / pixel_num * 4
            mse = (4 * mse_y + mse_u + mse_v) / 6 * 3   # rgb is sum, not average MSE
        else:
            mse = self.mse(pred, target)
            mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
        mse_loss_lambda = torch.sum(mse*lambda_w) / len(lambda_w)
        return self.loss_weight*mse_loss_lambda
 
 