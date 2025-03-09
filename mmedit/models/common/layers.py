# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from torch import nn
import torch
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, degred_vc_dim=64, ratio=0.25, stride=1,
                  padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True,
                  xavier_normal=False, with_route=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.xavier_normal=xavier_normal
        self.with_route=with_route
        if with_route:
            self.routing = nn.Sequential(
                nn.Linear(256, degred_vc_dim),
                nn.LeakyReLU(0.1, True),
                nn.Linear(degred_vc_dim, K),
                nn.Softmax(1)
            )
 
        self.weight_pool = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias_pool = None

        if xavier_normal:
            self._xavier_initialize_weights()
        else:
            self._kaiming_initialize_weights()

    def _kaiming_initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.weight_pool[i], mode='fan_out')
            if self.bias:
                nn.init.constant_(self.bias_pool[i], 0.)
     
    # def _kaiming_initialize_weights(self):
    #     for i in range(self.K):
    #         nn.init.kaiming_normal_(self.weight_pool[i], mode='fan_out')
    #         if self.bias:
    #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.bias_pool[i])
    #             bound = 1 / math.sqrt(fan_in)
    #             nn.init.uniform_(self.bias_pool[i], -bound, bound)


    def _xavier_initialize_weights(self):
        for i in range(self.K):
            nn.init.xavier_normal_(self.weight_pool[i], 1.)
            if self.bias:
                nn.init.constant_(self.bias_pool[i], 0)
    
    def forward(self, x, degrated_vc): 
        route=self.routing(degrated_vc) if self.with_route else degrated_vc
        batch_size, in_planes, height, width = x.size()
        weight = self.weight_pool.view(self.K, -1)

        aggregate_weight = torch.mm(route, weight).view(self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias:
            aggregate_bias = torch.mm(route, self.bias_pool).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
    

    
class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=stride)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.LeakyReLU(inplace=inplace),
        )

    def forward(self, x):
        x = self.down(x)
        identity = x
        out = self.conv(x)
        out = out + identity
        return out

class ResidualBlockWithStride2_dynamic(ResidualBlockWithStride2):
    def __init__(self, in_ch, out_ch, stride=2, inplace=False, num_expert=1, degred_vc_dim=32, 
                 xavier_normal=False, with_route=True):
        super().__init__(in_ch, out_ch, stride=stride, inplace=inplace)
        self.degred_conv = Dynamic_conv2d(out_ch, out_ch, 3, padding=1, K=num_expert, degred_vc_dim=degred_vc_dim, 
                                          xavier_normal=xavier_normal,with_route=with_route)
 
    def forward(self, x, degred_vc):
        
        x = self.down(x)
        identity = x
        out_degred=self.degred_conv(x, degred_vc)
        out = self.conv(out_degred)
        out = out + identity
        return out
 

 
class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class DepthConv2(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        )
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.out_conv(x1 * x2)
        return identity + x


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class ConvFFN2(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)


class ConvFFN3(nn.Module):
    def __init__(self, in_ch, inplace=False):
        super().__init__()
        expansion_factor = 2
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = self.relu1(x1) + self.relu2(x2)
        return identity + self.conv_out(out)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN2(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock3(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv2(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN2(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock4(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN3(out_ch, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)
