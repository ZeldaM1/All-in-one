# Copyright (c) OpenMMLab. All rights reserved.

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.registry import BACKBONES
from .restormer_net import TransformerBlock,Downsample,Upsample,LayerNorm
from einops import rearrange
from mmengine.model import BaseModule
from .restormer_net import OverlapPatchEmbed,TransformerBlock,Downsample,Upsample

from omegaconf import OmegaConf
import numpy as np
from .convmod import Ghost_Spatial_Attn
 
  
    
 
class TransformerBlockv4(TransformerBlock): 
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, residual=False, sigmoid=True,softmax=False,**kwrags):
        super(TransformerBlockv4, self).__init__(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, residual=residual)
        self.attn_s=Ghost_Spatial_Attn(dim, sigmoid,softmax) 
        self.norm3 = LayerNorm(dim, LayerNorm_type)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.attn_s(self.norm3(x))
        x = x + self.ffn(self.norm2(x))
        return x
 

def get_transformerblock(version): 
    if version=='v4':
        return TransformerBlockv4
    else:
        assert NotImplementedError
 
class Downformer_Blockv2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor,bias, LayerNorm_type, num_blocks=1,residual=False, version='v2',**kwrags):
        super().__init__()
        blocks=[]
        tranformer_block=get_transformerblock(version)
        for i in range(num_blocks):
            transblock=tranformer_block(dim=dim,num_heads=num_heads,ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias,LayerNorm_type=LayerNorm_type,residual=residual,**kwrags)
            blocks.append(transblock)
    
        blocks.append(Downsample(dim))
        self.blocks = nn.Sequential(*blocks)
 
    def forward(self, fea):
        return self.blocks(fea)

class Upformer_Blockv2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor,bias, LayerNorm_type, num_blocks=1, residual=False, version='v2',**kwrags):
        super().__init__()
        blocks=[Upsample(dim*2)]
        tranformer_block=get_transformerblock(version)
        for i in range(num_blocks):
            transblock=tranformer_block(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type, residual=residual,**kwrags)
            blocks.append(transblock)
        self.blocks = nn.Sequential(*blocks)
 
    def forward(self, fea):
        return self.blocks(fea)

 
 
 
@BACKBONES.register_module()
class Restor_Encoderv2_pos3(nn.Module):
    def __init__(self, dim, heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, residual=False, version='v2',input_resolution=0,**kwargs):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(3, dim)
        self.encoder_x2 = Downformer_Blockv2(dim,heads[0],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[0],residual=residual, version=version, input_resolution=input_resolution,**kwargs)
        self.encoder_x4 = Downformer_Blockv2(dim*2,heads[1],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[1],residual=residual, version=version, input_resolution=input_resolution//2,**kwargs)
        self.encoder_x8 = Downformer_Blockv2(dim*4,heads[2],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[2],residual=residual, version=version, input_resolution=input_resolution//4,**kwargs)
        self.encoder_x16 = Downformer_Blockv2(dim*8,heads[3],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[3],residual=residual, version=version, input_resolution=input_resolution//8,**kwargs)
    def forward(self, x, quant_step): 
        embed_fea = self.patch_embed(x)
        feax2 = self.encoder_x2(embed_fea)
        feax4 = self.encoder_x4(feax2)
        feax4 = feax4 * quant_step
        feax8 = self.encoder_x8(feax4)
        return self.encoder_x16(feax8)

 
@BACKBONES.register_module()
class Restor_Decoderv2_pos3(nn.Module):
    def __init__(self, dim, heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, residual=False, version='v2',input_resolution=None,**kwargs):
        super().__init__()
        self.decoder_x8 = Upformer_Blockv2(dim*8,heads[3],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[3],residual=residual, version=version, input_resolution=input_resolution,**kwargs)
        self.decoder_x4 = Upformer_Blockv2(dim*4,heads[2],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[2],residual=residual, version=version, input_resolution=input_resolution*2,**kwargs)
        self.decoder_x2 = Upformer_Blockv2(dim*2,heads[1],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[1],residual=residual, version=version, input_resolution=input_resolution*4,**kwargs)
        self.decoder_x1 = Upformer_Blockv2(dim,heads[0],ffn_expansion_factor,bias,LayerNorm_type,num_blocks[0],residual=residual, version=version, input_resolution=input_resolution*8,**kwargs)
    
    def forward(self, x, quant_step): 
        feax8 = self.decoder_x8(x)
        feax4 = self.decoder_x4(feax8)
        feax4 = feax4 * quant_step
        feax2 = self.decoder_x2(feax4)
        return self.decoder_x1(feax2)
  
 

 