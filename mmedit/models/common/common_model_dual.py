# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn

from pytorch_msssim import MS_SSIM

from .video_net import LowerBound
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
 
from mmedit.utils import get_padding_size, yuv_444_to_420
 

class CompressionModel_dual(nn.Module):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__()

        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(z_channel)
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(mv_z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.force_zero_thres = None
        self.noise_level = 0.5
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.mse = nn.MSELoss(reduction='none')
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)

        self.masks = {}
        self.force_generate_mask = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

 
    def quant(self, x):
        return torch.round(x)

    def get_curr_q(self, q_scale, q_basic, *, batch_size=1, q_index=None):
        if isinstance(q_index, list):
            anchor_num = len(q_index)
            min_q = q_scale[0:1, :, :, :]
            max_q = q_scale[1:2, :, :, :]
            step = (torch.log(max_q) - torch.log(min_q)) / (self.get_qp_num() - 1)
            q_scale = [torch.exp(torch.log(min_q) + step * i) for i in q_index]
            q_scale = torch.cat(q_scale, dim=0)
            q_basic = q_basic.repeat(batch_size * anchor_num, 1, 1, 1)
            q_scale = q_scale.repeat_interleave(batch_size, dim=0)
        else:
            q_scale = q_scale[q_index]

        return q_basic * q_scale

    @staticmethod
    def get_qp_num():
        return 64
 
    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)


    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)
 
    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    def get_mse(self, x, x_hat, yuv420):
        _, _, H, W = x.size()
        pixel_num = H * W
        mse = self.mse(x, x_hat)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
        return mse

    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)
 
    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
         
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat
 
    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks or self.force_generate_mask:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]


    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = LowerBound.apply(bits, 0)
        return bits


    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        probs = probs.to(torch.float32)
        return CompressionModel_dual.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        probs = probs.to(torch.float32)
        return CompressionModel_dual.probs_to_bits(probs)

    def forward_dual_prior(self, y, common_params, y_spatial_prior, y_spatial_prior_reduction=None, write=False):
 
        quant_step, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
 
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, dtype, device)
 
        quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step
 
        y_0, y_1 = y.chunk(2, 1)
        scales_0, scales_1 = scales.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)
        y_res_0_0, y_q_0_0, y_hat_0_0, scales_hat_0_0 =  self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, scales_hat_1_1 =  self.process_with_mask(y_1, scales_1, means_1, mask_1)
        params = torch.cat((y_hat_0_0, y_hat_1_1, common_params), dim=1)
        scales_0, means_0, scales_1, means_1 = y_spatial_prior(params).chunk(4, 1)

        y_res_0_1, y_q_0_1, y_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)

        y_res_0 = y_res_0_0 + y_res_0_1
        y_q_0 = y_q_0_0 + y_q_0_1
        y_hat_0 = y_hat_0_0 + y_hat_0_1
        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1

        y_res_1 = y_res_1_1 + y_res_1_0
        y_q_1 = y_q_1_1 + y_q_1_0
        y_hat_1 = y_hat_1_1 + y_hat_1_0
        scales_hat_1 = scales_hat_1_1 + scales_hat_1_0

        y_res = torch.cat((y_res_0, y_res_1), dim=1)
        y_q = torch.cat((y_q_0, y_q_1), dim=1)
        y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        scales_hat = torch.cat((scales_hat_0, scales_hat_1), dim=1)

        y_hat = y_hat * quant_step

        if write:
            y_q_w_0 = y_q_0_0 + y_q_1_1
            y_q_w_1 = y_q_0_1 + y_q_1_0
            scales_w_0 = scales_hat_0_0 + scales_hat_1_1
            scales_w_1 = scales_hat_0_1 + scales_hat_1_0
            return y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat
        return y_res, y_q, y_hat, scales_hat

 