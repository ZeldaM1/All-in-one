from torch import nn
from .deform_conv import DCN_layer
from .moco import MoCo
from mmedit.models.registry import BACKBONES,MODELS
from mmedit.models.restorers import BasicRestorer
from mmedit.models.builder import build_backbone, build_loss
import torch
from mmedit.utils import get_root_logger
from mmcv.runner import load_checkpoint
import os.path as osp
import mmcv
from mmedit.core import psnr, ssim, tensor2img
import numbers

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter


class CBDE(nn.Module):
    def __init__(self, batch_size):
        super(CBDE, self).__init__()

        dim = 256

        # Encoder
        self.E = MoCo(base_encoder=ResEncoder, dim=dim, K=batch_size * dim)

    def forward(self, x_query, x_key):
        if self.training:
            # degradation-aware represenetion learning
            fea, logits, labels, inter = self.E(x_query, x_key)

            return fea, logits, labels, inter
        else:
            # degradation-aware represenetion learning
            fea, inter = self.E(x_query, x_query)
            return fea, inter



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(DGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter)
        out = dcn_out + sft_out
        out = x + out

        return out


class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(n_feat, n_feat, kernel_size)
        self.dgm2 = DGM(n_feat, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter))
        out = self.conv2(out) + x

        return out


class DGG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks):
        super(DGG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGB(conv, n_feat, kernel_size) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        return res


class DGRN(nn.Module):
    def __init__(self, conv=default_conv):
        super(DGRN, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # body
        modules_body = [
            DGG(default_conv, n_feats, kernel_size, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter):
        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x



@BACKBONES.register_module()
class AirNet(nn.Module):
    def __init__(self, batch_size):
        super(AirNet, self).__init__()

        # Restorer
        self.R = DGRN()

        # Encoder
        self.E = CBDE(batch_size)

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
        

    def forward(self, x_query, x_key=None, encoder_only=False):
        if self.training:
            if encoder_only:
                return self.E(x_query, x_key)
 
            fea, logits, labels, inter = self.E(x_query, x_key)
            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter)

            return restored

@MODELS.register_module()
class AirNet_BasicRestorer(BasicRestorer):
    def __init__(self,
                 generator,
                 pixel_loss,
                 contrast_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator,pixel_loss,train_cfg,test_cfg,pretrained)
        self.enc_iter = train_cfg.get('enc_iter', 0) if train_cfg else 0
        self.contrast_loss = build_loss(contrast_loss)
        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))
 

    def forward_train(self, lq, gt):
        losses = dict()
        degrad_patch_1, degrad_patch_2=lq
        clean_patch_1, clean_patch_2=gt

        # mmcv.imwrite(tensor2img(degrad_patch_1[0:1]), './degrad_patch_1.png')
        # mmcv.imwrite(tensor2img(clean_patch_1[0:1]), './clean_patch_1.png')
        # mmcv.imwrite(tensor2img(degrad_patch_2[0:1]), './degrad_patch_2.png')
        # mmcv.imwrite(tensor2img(clean_patch_2[0:1]), './clean_patch_2.png')
        
 
        if self.step_counter < self.enc_iter:
            _, output, target, _ = self.generator(degrad_patch_1, degrad_patch_2, encoder_only=True)
            contrast_loss = self.contrast_loss(output, target)
            losses['loss_contrast'] = contrast_loss
        else:
            restored, output, target = self.generator(degrad_patch_1, degrad_patch_2)
            contrast_loss = self.contrast_loss(output, target)
            losses['loss_contrast'] = contrast_loss
            l1_loss = self.pixel_loss(restored, clean_patch_1)
            losses['loss_pix'] = l1_loss
            
 
        outputs = dict(
            losses=losses,
            num_samples=len(clean_patch_1.data)
            )
        return outputs

 
    def train_step(self, data_batch, optimizer):
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()
        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def forward_test(self,
                    lq,
                    gt=None,
                    meta=None,
                    save_image=False,
                    save_path=None,
                    iteration=None):

        output = self.generator(lq,lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                    f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results