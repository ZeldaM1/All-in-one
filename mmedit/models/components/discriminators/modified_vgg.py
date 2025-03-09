# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@COMPONENTS.register_module()
class ModifiedVGG(nn.Module):
    """A modified VGG discriminator with input size 128 x 128.

    It is used to train SRGAN and ESRGAN.

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        mid_channels (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, in_channels, mid_channels, in_size):
        super().__init__()
        
        self.conv0_0 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True)

        if in_size>128:
            self.conv0_1_add = nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=False)
            self.bn0_1_add = nn.BatchNorm2d(mid_channels, affine=True)

        self.conv0_1 = nn.Conv2d(
            mid_channels, mid_channels, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(mid_channels, affine=True)

        self.conv1_0 = nn.Conv2d(
            mid_channels, mid_channels * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(mid_channels * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            mid_channels * 2, mid_channels * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(mid_channels * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            mid_channels * 2, mid_channels * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(mid_channels * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            mid_channels * 4, mid_channels * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(mid_channels * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            mid_channels * 4, mid_channels * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(mid_channels * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(mid_channels * 8, affine=True)

        self.conv4_0 = nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(mid_channels * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(mid_channels * 8, affine=True)

        self.linear1 = nn.Linear(mid_channels * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        # assert x.size(2) == 128 and x.size(3) == 128, (
        #     f'Input spatial size must be 128x128, '
        #     f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        if hasattr(self,"conv0_1_add"):
            feat = self.lrelu(self.bn0_1_add(self.conv0_1_add(feat))) 


        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # Use PyTorch default initialization.
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')




class SinConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride= 1,padding= 0,dilation=1,groups=1,
                 bias='auto',norm_cfg='BN',act_cfg='ReLU', inplace=True, padding_mode='zeros',
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.order=order

        conv_bias=False if self.with_norm else bias
        # build convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=conv_bias,padding_mode=padding_mode)
   
        # build normalization layers
        if self.with_norm:
            if norm_cfg=='BN':
                self.norm = nn.BatchNorm2d(out_channels, affine=True)
            else:
                assert NotImplementedError

        # build activation layer
        if self.with_activation:
            if act_cfg=='LeakyReLU':
                self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
            elif act_cfg=='ReLU':
                self.activate = nn.ReLU(inplace=inplace)
            else:
                assert NotImplementedError
    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x
            
 
@COMPONENTS.register_module()
class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 base_channels,
                 min_feat_channels,
                 kernel_size,
                 padding,
                 num_layers,
                 norm_cfg='BN',
                 act_cfg='LeakyReLU',
                 stride=1,
                 **kwargs):
        super().__init__()

        self.base_channels = base_channels
        self.stride = stride
        self.head = SinConv(
            in_channels,
            base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.body = nn.Sequential()

        for i in range(num_layers - 2):
            feat_channels_ = int(base_channels / pow(2, (i + 1)))
            block = SinConv(
                max(2 * feat_channels_, min_feat_channels),
                max(feat_channels_, min_feat_channels),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs)
            self.body.add_module(f'block{i+1}', block)

        self.tail = SinConv(
            max(feat_channels_, min_feat_channels),
            1,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            norm_cfg=None,
            act_cfg=None,
            **kwargs)

        self.init_weights()

    def forward(self, x):
 
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x

    # TODO: study the effects of init functions
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                constant_init(m, 1)
 



