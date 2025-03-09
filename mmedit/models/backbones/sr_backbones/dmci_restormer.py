# Copyright (c) OpenMMLab. All rights reserved.
from logging import WARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.registry import BACKBONES
from mmedit.models.common import conv3x3, DepthConvBlock2,DepthConvBlock4,ResidualBlockUpsample,CompressionModel_dual
from mmedit.utils import encode_i, decode_i, get_downsampled_shape, filesize,get_state_dict 
import time
from .restormer_net import TransformerBlock


@BACKBONES.register_module()
class Hyper_enc(nn.Module):
    def __init__(self, N, z_channel, inplace):
        super().__init__()
        self.blocks = nn.Sequential(
            DepthConvBlock4(N, z_channel, inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )
    def forward(self, y_pad):
        return self.blocks(y_pad)
@BACKBONES.register_module()
class Hyper_dec(nn.Module):
    def __init__(self, N, z_channel, inplace):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )
    def forward(self, z_hat):
        return self.blocks(z_hat)

 
 
@BACKBONES.register_module()
class DMCI_ED_restormer_dual_redu(CompressionModel_dual):
    def __init__(self, dim=48, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree', num_blocks=[4, 6, 6, 8], num_refinement_blocks=None, residual=False,
                ec_thread=False, stream_part=1, inplace=False, yuv420=False,
                hyper_enc="Hyper_enc", hyper_dec="Hyper_dec", is_encoder=False,recon_only=False,
                intraEncoder="Restor_Encoder", version_enc=None, intraDecoder="Restor_Decoder", version_dec=None, z_ch=None, N_ch=None,
                input_resolution=0, split_size=None,N_ch_new=None, N_ch_newv2=None, lq_ratio=None,extend_ratio=None,sigmoid=True,fix_ed=False,
                ):
        z_channel=dim*2 if (not z_ch) else z_ch
        N_dim=dim*(2**4) if (not N_ch) else N_ch
        super().__init__(y_distribution='gaussian', z_channel=z_channel, ec_thread=ec_thread, stream_part=stream_part)
        self.yuv420=yuv420
        self.is_encoder=is_encoder
        self.recon_only=recon_only
        self.fix_ed=fix_ed
        self.enc = BACKBONES.build(dict(type=intraEncoder, dim=dim, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, 
                                        num_blocks=num_blocks,residual=residual, version=version_enc,input_resolution=input_resolution,split_size=split_size,
                                        lq_ratio=lq_ratio,extend_ratio=extend_ratio, sigmoid=sigmoid))
        self.dec = BACKBONES.build(dict(type=intraDecoder, dim=dim, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, 
                                        num_blocks=num_blocks,residual=residual, version=version_dec,input_resolution=input_resolution,split_size=split_size,
                                        lq_ratio=lq_ratio,extend_ratio=extend_ratio, sigmoid=sigmoid))
        if N_ch and (N_ch_new is None):
            self.N_dim_compress = nn.Sequential(nn.Conv2d(dim*(2**4), N_dim, 3, 1, 1), nn.ReLU(inplace=True)) 
            self.N_dim_decompress = nn.Sequential(nn.Conv2d(N_dim, dim*(2**4), 3, 1, 1, groups=N_dim, bias=False), nn.ReLU(inplace=True)) 
        if N_ch_new:
            self.N_dim_compress =  nn.Conv2d(dim*(2**4), N_dim, 3, 1, 1) 
            self.N_dim_decompress = nn.Conv2d(N_dim, dim*(2**4), 3, 1, 1)
        
          
        self.hyper_enc = BACKBONES.build(dict(type=hyper_enc, N=N_dim, z_channel=z_channel, inplace=inplace)) 
        self.hyper_dec = BACKBONES.build(dict(type=hyper_dec, N=N_dim, z_channel=z_channel, inplace=inplace)) 

        self.y_prior_fusion = nn.Sequential(DepthConvBlock4(N_dim, N_dim * 2, inplace=inplace), DepthConvBlock4(N_dim * 2, N_dim * 3, inplace=inplace),)
        self.y_spatial_prior_reduction = nn.Conv2d(N_dim * 3, N_dim * 1, 1)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N_dim * 2, N_dim * 2, inplace=inplace),
            DepthConvBlock2(N_dim * 2, N_dim * 2, inplace=inplace),
            DepthConvBlock2(N_dim * 2, N_dim * 2, inplace=inplace),
        )
 
        if num_refinement_blocks:
            refine=[]
            for i in range(num_refinement_blocks):
                refine.append(TransformerBlock(
                dim=int(dim),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                residual=residual))
            refine.append(conv3x3(dim, 3))
            self.refine=nn.Sequential(*refine)
            
        else:
            self.refine = conv3x3(dim, 3)
        self.q_basic_enc = nn.Parameter(torch.ones((1, z_channel, 1, 1)))
        self.q_scale_enc = nn.Parameter(torch.ones((2, 1, 1, 1)))
        self.q_scale_enc_fine = torch.ones((self.get_qp_num(),))
        self.q_basic_dec = nn.Parameter(torch.ones((1, z_channel, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((2, 1, 1, 1)))
        self.q_scale_dec_fine = torch.ones((self.get_qp_num(),))

        self.N = int(N_dim)
        self._initialize_weights()

 

    def get_q_for_inference(self, q_index):
        curr_q_enc = self.get_curr_q(self.q_scale_enc_fine, self.q_basic_enc, q_index=q_index)
        curr_q_dec = self.get_curr_q(self.q_scale_dec_fine, self.q_basic_dec, q_index=q_index)
        return curr_q_enc, curr_q_dec

    def forward(self, x, gt=None, q_index=None,calc_ssim=False, recon_only=False, is_encoder=False):
       
        B, _, _, _ = x.size()
        if isinstance(q_index, list):
            curr_q_enc = self.get_curr_q(self.q_scale_enc, self.q_basic_enc,  q_index=q_index, batch_size=B)
            curr_q_dec = self.get_curr_q(self.q_scale_dec, self.q_basic_dec, q_index=q_index, batch_size=B)
            anchor_num = len(q_index)
            x = x.repeat(anchor_num, 1, 1, 1)
            gt = gt.repeat(anchor_num, 1, 1, 1)
        else:
            assert B == 1
            curr_q_enc, curr_q_dec = self.get_q_for_inference(q_index)
        if self.fix_ed:
            with torch.no_grad():
                y = self.enc(x, curr_q_enc) 
        else:
            y = self.enc(x, curr_q_enc) 
 
        if hasattr(self,'N_dim_compress'):
            y=self.N_dim_compress(y)
        if self.is_encoder or is_encoder:
            return None
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = self.quant(z)
  
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
  
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(y, params, self.y_spatial_prior, y_spatial_prior_reduction=self.y_spatial_prior_reduction)
        if hasattr(self,'N_dim_decompress'):
            y_hat=self.N_dim_decompress(y_hat)
   
        x_latent = self.dec(y_hat, curr_q_dec)
        x_hat=self.refine(x_latent) 
    
        if self.recon_only or recon_only:
            mse = self.get_mse(gt, x_hat, yuv420=self.yuv420)  
            if calc_ssim:
                ssim = self.ssim(gt, x_hat)
            else:
                ssim = torch.zeros_like(mse)
 
            return {"x_hat": x_hat, "mse": mse, "ssim": ssim}
 
        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        mse = self.get_mse(gt, x_hat, yuv420=self.yuv420)  
        if calc_ssim:
            ssim = self.ssim(gt, x_hat)
        else:
            ssim = torch.zeros_like(mse)

        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z
 
        return {"x_hat": x_hat, "bit": bits,"bpp": bpp,"bpp_y": bpp_y,"bpp_z": bpp_z, "mse": mse, "ssim": ssim}


    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scale_enc = ckpt["q_scale_enc"].reshape(-1)
        q_scale_dec = ckpt["q_scale_dec"].reshape(-1)
        return q_scale_enc, q_scale_dec

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              all_missing_keys, unexpected_keys,err_msg):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      all_missing_keys, unexpected_keys,err_msg)

        with torch.no_grad():
            q_scale_enc_fine = torch.linspace(torch.log(self.q_scale_enc[0, 0, 0, 0]), torch.log(self.q_scale_enc[1, 0, 0, 0]), self.get_qp_num())
            self.q_scale_enc_fine = torch.exp(q_scale_enc_fine)
            q_scale_dec_fine = torch.linspace(torch.log(self.q_scale_dec[0, 0, 0, 0]), torch.log(self.q_scale_dec[1, 0, 0, 0]), self.get_qp_num())
            self.q_scale_dec_fine = torch.exp(q_scale_dec_fine)

 
 
 
@BACKBONES.register_module()
class DMCI_ED_restormer_dual(DMCI_ED_restormer_dual_redu):
    def __init__(self, dim=48, N_ch=None,  inplace=False, **kwargs):
        N_dim=dim*(2**4) if (not N_ch) else N_ch
        super().__init__(dim=dim, N_ch=N_ch,  inplace=inplace,**kwargs)
       
        self.y_spatial_prior_reduction = None
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N_dim * 4, N_dim * 3, inplace=inplace),
            DepthConvBlock2(N_dim * 3, N_dim * 2, inplace=inplace),
            DepthConvBlock2(N_dim * 2, N_dim * 2, inplace=inplace),
        )
 
        self._initialize_weights()

  