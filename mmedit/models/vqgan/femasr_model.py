from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from ..registry import MODELS 
import torch
import torchvision.utils as tvu
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from mmedit.models.restorers import SRGAN
import copy
import os
from mmedit.utils import generate_str
from mmedit.core import tensor2img
from mmedit.apis.test import PSNR
import numpy as np
import mmcv
import numbers
from mmedit.utils.functional import ycbcr444_to_420,ycbcr420_to_rgb



def generate_strv2(x):
    return f'{x.item():.5f} '


@MODELS.register_module()
class FeMaSRModel_HR(SRGAN): # stage 1 
    def __init__(self, generator, discriminator=None, codebook_loss=None,semantic_loss=None, input_format='RGB',**kwargs):
        super().__init__(generator, discriminator=discriminator, **kwargs)
        self.codebook_loss = build_loss(codebook_loss) if (codebook_loss is not None) else None
        self.semantic_loss = build_loss(semantic_loss) if (semantic_loss is not None) else None
        self.input_format=input_format
  
    def train_step(self, input_data, optimizer):
        gt=input_data['gt']
        
        # gt = input_data['gt'] if input_data isidct
        output, l_codebook, l_semantic, _ = self.generator(gt) 
 
        losses = dict()
        log_vars = dict()
        set_requires_grad(self.discriminator, False)
        
        if (self.step_counter % self.disc_steps == 0 and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(output, gt)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style
            if self.codebook_loss:
                losses['loss_codebook'] = self.codebook_loss(l_codebook)
            if self.semantic_loss:
                losses['loss_semantic'] = self.semantic_loss(l_semantic)
            
            # gan loss for generator
            fake_g_pred = self.discriminator(output)
            losses['loss_gan'] = self.gan_loss(fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            outputs=dict()
            if skip_batch(self.generator):
                outputs["skip_batch"]=True
                return outputs
            else:
                optimizer['generator'].step()

         # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.gan_loss(real_d_pred, target_is_real=True, is_disc=True)
        # fake
        fake_d_pred = self.discriminator(output.detach())
        loss_d_fake = self.gan_loss(fake_d_pred, target_is_real=False, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_real=loss_d_real,loss_d_fake=loss_d_fake))
        log_vars.update(log_vars_d)
        # optimize
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            info=None,
            skip_batch=False)
 
        return outputs 
 
    def forward(self, gt, test_mode=False, meta=None, save_image=False, save_path=None, iteration=None,**kwargs):
        if test_mode:
            min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
            _, _, h, w = gt.shape
            use_title=False if h*w < min_size else True
            output=self.generator(gt,test_mode=True,use_title=use_title)
        else:
            raise ValueError('follow SRGAN, we do not have forward_train function')
        
 
        img_name = os.path.basename(meta[0]['gt_path']).split('.png')[0]
        if output.ndim == 4:  # an image, key = 000001/0000 (Vimeo-90K)
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, f'{img_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{img_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                f'but got {type(iteration)}')
           
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, ('evaluation with metrics must have gt images.')
            if self.input_format=='YUV':
                evaluate_results,(y_rec,uv_rec) = self.evaluate(output, gt, True)  
            else:
                evaluate_results = self.evaluate(output, gt, False)  
            results = dict(eval_result=evaluate_results)
 
            if save_image:
                if self.input_format=='YUV':
                        x_hat_rgb = ycbcr420_to_rgb(y_rec,uv_rec, order=1).transpose(1, 2, 0)
                        x_hat_rgb = np.clip(np.rint(x_hat_rgb * 255), 0, 255).astype(np.uint8)
                        x_hat_rgb = x_hat_rgb[:,:,[2,1,0]]
                        mmcv.imwrite(x_hat_rgb, save_path)
                elif self.input_format=='RGB':
                    mmcv.imwrite(tensor2img(output), save_path) 
            
            return results
        else:
            return output
 
    def evaluate(self, output, gt, return_yuv=False):
        eval_result = dict()
        if self.input_format=='RGB':
            crop_border = self.test_cfg.crop_border
            convert_to = self.test_cfg.get('convert_to', None)
            for metric in self.test_cfg.metrics:
                if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                    avg = []
                    for i in range(0, output.size(1)):
                        output_i = tensor2img(output[:, i, :, :, :])
                        gt_i = tensor2img(gt[:, i, :, :, :])
                        avg.append(self.allowed_metrics[metric](
                            output_i, gt_i, crop_border, convert_to=convert_to))
                    eval_result[metric] = np.mean(avg)
                elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
                    output_img = tensor2img(output)
                    gt_img = tensor2img(gt)
                    value = self.allowed_metrics[metric](
                        output_img, gt_img, crop_border, convert_to=convert_to)
                    eval_result[metric] = value
            return eval_result
        elif self.input_format=='YUV':
            value,y,uv=PSNR(output, gt)
            eval_result['PSNR'] = value
            if return_yuv:
                return eval_result,(y,uv)
            else:
                return eval_result
        else:
            assert NotImplementedError
        


    def vis_single_code(self, up_factor=2):
        generator = self.get_bare_model(self.generator)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = generator.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

 
 
 