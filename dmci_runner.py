# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check
from torch.utils.data import DataLoader
import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner import EpochBasedRunner,IterBasedRunner, IterLoader, get_host_info
import random
from mmedit.utils import get_training_lambdas
import torch
import time

@RUNNERS.register_module()
class DMCI_EpochBasedRunner(EpochBasedRunner):
    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            training_strategy_all=None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for curren_epoch in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], training_strategy_all[curren_epoch][2:5], **kwargs)

    def train(self, data_loader,  training_strategy=None, lmbdas=None,**kwargs):
        device = next(self.model.parameters()).device
 
        loss_type,is_rand_rate,rate_num = training_strategy
        
        training_lmbdas, q_indexes = get_training_lambdas(lmbdas, rate_num,self.model.module.generator.module.get_qp_num(), is_rand_rate, device)
        iter_stop = 4 // rate_num

    
        self.model.train()
        # breakpoint()
        
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        # torch.cuda.synchronize()
        # t0 = time.time()
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch 
            self._inner_iter = i
            self.call_hook('before_train_iter')
            # print(iter_stop)

            for it in range(iter_stop):
                print_loss_info = (i % 100 == 0)

                if is_rand_rate:
                    q_index = [random.randint(0,self.model.module.generator.module.get_qp_num() - 1) for _ in range(rate_num)]
                    curr_lmbdas = torch.tensor([training_lmbdas[i] for i in q_index],
                                            dtype=torch.float32, device=device)
                else:
                    q_index = q_indexes[it]
                    curr_lmbdas = training_lmbdas[it]
                
                self.run_iter(data_batch["gt"].to(device), train_mode=True, q_index=q_index, lmbdas=curr_lmbdas, loss_type=loss_type, get_loss_info=print_loss_info,**kwargs)
                
 
                if print_loss_info:
                    print(self.outputs["info"])
                    # torch.cuda.synchronize()
                    # t1 = time.time()
                    # print(f" time: {t1-t0:.3f} seconds")
                    # if it == iter_stop - 1:
                    #     t0 = t1
            
            
            self.call_hook('after_train_iter')

            del self.data_batch
            self._iter += 1
 
        self.call_hook('after_train_epoch')
        self._epoch += 1



@RUNNERS.register_module()
class DMCI_IterBasedRunner(IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, training_strategy=None, lmbdas=None, print_loss_info=None,**kwargs):
        device = next(self.model.parameters()).device
        loss_type,is_rand_rate,rate_num = training_strategy
        training_lmbdas, q_indexes = get_training_lambdas(lmbdas, rate_num,self.model.module.generator.module.get_qp_num(), is_rand_rate, device)
        iter_stop = 4 // rate_num

        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')

        for it in range(iter_stop):
            
            if is_rand_rate:
                q_index = [random.randint(0,self.model.module.generator.module.get_qp_num() - 1) for _ in range(rate_num)]
                curr_lmbdas = torch.tensor([training_lmbdas[i] for i in q_index], dtype=torch.float32, device=device)
            else:
                q_index = q_indexes[it]
                curr_lmbdas = training_lmbdas[it]
            outputs = self.model.train_step(data_batch, self.optimizer, q_index=q_index, lmbdas=curr_lmbdas, 
                                            loss_type=loss_type, get_loss_info=print_loss_info, **kwargs)
            if outputs["skip_batch"]:
                continue

            if print_loss_info and (outputs["info"] is not None):
                print(outputs["info"])
        
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1


    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_iters: Optional[int] = None,
            training_strategy_map=None,
            **kwargs) -> None:
 
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    print_loss_info = (self.iter % 2000 == 0)
                    # print(self.iter, print_loss_info,training_strategy_map(self.iter))
                    iter_runner(iter_loaders[i], training_strategy_map(self.iter), print_loss_info=print_loss_info,**kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')
        
 