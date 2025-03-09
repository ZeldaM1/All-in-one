# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import is_module_wrapper

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer

@MODELS.register_module()
class RealESRGAN(BasicRestorer):
    def __init__(self,
                 generator,
                 pixel_loss=None,
                 perceptual_loss=None,
                 is_use_ema=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__(generator, pixel_loss, train_cfg, test_cfg, pretrained)

        self.is_use_ema = is_use_ema
        if is_use_ema:
            self.generator_ema = deepcopy(self.generator)
        else:
            self.generator_ema = None

        del self.step_counter
        self.register_buffer('step_counter', torch.zeros(1))

        if train_cfg is not None:  # used for initializing from ema model
            self.start_iter = train_cfg.get('start_iter', -1)
        else:
            self.start_iter = -1

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())

        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

 
        # generator
        fake_g_output = self.generator(lq)

        losses = dict()
        log_vars = dict()

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        _model = self.generator_ema if self.is_use_ema else self.generator
        output = _model(lq)

        if self.test_cfg is not None and self.test_cfg.get(
                'metrics', None) and gt is not None:
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())

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
