# Copyright (c) OpenMMLab. All rights reserved.
from logging import WARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import MODELS 
import numpy as np
from mmedit.utils import avg_per_rate, generate_str
from collections import OrderedDict
import time 
from mmedit.utils import get_clip_grad_norm_func, get_state_dict

grad_clip=get_clip_grad_norm_func(True)

def skip_batch(model):
    skip_flag=False
    total_norm = grad_clip(model.parameters(), 0.1)
    if total_norm.isnan() or total_norm.isinf():
        print("non-finite norm, skip this batch")
        skip_flag=True
    return skip_flag
    


@MODELS.register_module()
class DMCI(nn.Module):
    def __init__(self, generator, RD_loss=None, 
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 fix_bpp=False,
                 fix_ed=False,
                 ):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # # generator
        self.generator = MODELS.build(generator)
        # loss
        self.RD_loss = MODELS.build(RD_loss) if (RD_loss is not None) else None
        self.step_counter = 0
        if fix_bpp:
            self.generator.requires_grad_(False)
            self.generator.enc.requires_grad_(True)
            self.generator.dec.requires_grad_(True)
        if fix_ed:
            self.generator.requires_grad_(True)
            self.generator.enc.requires_grad_(False)
            self.generator.dec.requires_grad_(False)

    def get_rd_info(self, result, batch_size, anchor_num):
        rd={}
        for key in ["bpp_y","bpp_z","bpp","mse","ssim"]:
            if key in result.keys():
                rd[key]=avg_per_rate(result[key], batch_size, anchor_num)
        if "ssim" in rd.keys():
            rd["1-ssim"]=1-rd["ssim"]

 
        return rd

    @staticmethod
    def get_loss_info(rd, loss):
        info={}
        for key in ["bpp_y","bpp_z","mse","ssim"]:
            if key in rd.keys():
                info[key]= generate_str(rd[key])
        info["costs"]=generate_str(loss['costs']),
        info["losses"]= generate_str(loss['losses']),
      
        return info
    @staticmethod
    def get_anchor_num(q_index):
        if isinstance(q_index, list):
            return len(q_index)
        return 1
 
    @staticmethod
    def get_q_scales_from_ckpt(self, ckpt_path):
        return self.generator.get_q_scales_from_ckpt(ckpt_path)
 
    
    def forward(self, lq, gt=None, q_index=None, calc_ssim=False, recon_only=False, test_mode=False, lmbdas=None, 
                loss_type=None, get_loss_info=False,meta=None):
        return self.generator(lq, gt, q_index, recon_only)
         
 
  
    def parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars


