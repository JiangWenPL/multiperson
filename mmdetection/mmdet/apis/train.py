from __future__ import division

import os.path as osp
import re
from collections import OrderedDict

import torch
from torch import nn
from mmcv.runner import Runner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet import datasets
from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        Fp16OptimizerHook)
from mmdet.datasets import build_dataloader, build_dataloader_fuse
from mmdet.models import RPN
from mmdet.core.utils import AverageMeter
from mmdet.models.smpl_heads.smpl_common import Discriminator
from .env import get_root_logger
from .adv_runner import AdvRunner
from mmdetection.mmdet.models.utils.smpl_utils import batch_rodrigues
from copy import deepcopy

from functools import partial
from mmcv.parallel import collate
from tqdm import tqdm
import random
import numpy as np
import pickle
from mmcv.runner.checkpoint import load_checkpoint
from mmdetection.mmdet.models.losses.smpl_loss import batch_adv_disc_l2_loss, batch_encoder_disc_l2_loss, \
    adversarial_loss


def parse_losses(losses, tag_tail=''):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if not ('loss' in loss_name):
            losses[loss_name] = loss_value.cpu()

    if 'img$idxs_in_batch' in losses:
        last_idx = -1
        split_idx = -1
        for i, idx in enumerate(losses['img$idxs_in_batch'].squeeze()):
            if last_idx > idx:
                split_idx = i
                break
            else:
                last_idx = idx
        split_idx = int(split_idx)
        if split_idx > 0:
            for loss_name, loss_value in losses.items():
                if loss_name.startswith('img$') and loss_name != 'img$raw_images':
                    losses[loss_name] = losses[loss_name][:split_idx]

    for loss_name, loss_value in losses.items():
        # To avoid stats pollution for validation inside training epoch.
        loss_name = f'{loss_name}/{tag_tail}'
        if loss_name.startswith('img$'):
            log_vars[loss_name] = loss_value
            continue
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    # TODO: Make the code more elegant here.
    log_vars[f'loss/{tag_tail}'] = loss
    for name in log_vars:
        if not name.startswith('img$'):
            log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, mode, **kwargs):
    # NOTE: The mode is str instead of boolean now.
    losses = model(**data)
    tag_tail = mode
    loss, log_vars = parse_losses(losses, tag_tail)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def parse_adv_losses(losses, tag_tail=''):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if not ('loss' in loss_name):
            if isinstance(losses[loss_name], torch.Tensor):
                losses[loss_name] = loss_value.cpu()

    if 'img$idxs_in_batch' in losses:
        last_idx = -1
        split_idx = -1
        for i, idx in enumerate(losses['img$idxs_in_batch'].squeeze()):
            if last_idx > idx:
                split_idx = i
                break
            else:
                last_idx = idx
        split_idx = int(split_idx)
        if last_idx > 0:
            losses['img$raw_images'] = losses['img$raw_images'][:int(last_idx) + 1]
        if split_idx > 0:
            for loss_name, loss_value in losses.items():
                if loss_name.startswith('img$') and loss_name != 'img$raw_images':
                    losses[loss_name] = losses[loss_name][:split_idx]

    for loss_name, loss_value in losses.items():
        # To avoid stats pollution for validation inside training epoch.
        loss_name = f'{loss_name}/{tag_tail}'
        if loss_name.startswith('img$'):
            log_vars[loss_name] = loss_value
            continue
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key and not _key.startswith('adv'))
    adv_loss = sum(_value for _key, _value in log_vars.items() if _key.startswith('adv_loss'))
    # TODO: Make the code more elegant here.
    log_vars[f'loss/{tag_tail}'] = loss
    log_vars[f'adv_loss/{tag_tail}'] = adv_loss
    for name in log_vars:
        if not name.startswith('img$'):
            log_vars[name] = log_vars[name].item()

    return loss, adv_loss, log_vars


def adv_batch_processor(model, data, mode, **kwargs):
    # NOTE: The mode is str instead of boolean now.
    discriminator = kwargs.get('discriminator')
    re_weight = kwargs.get('re_weight', dict())
    losses = model(**data)
    pred_pose_shape = losses.pop('pred_pose_shape')
    mosh = kwargs.get('mosh')
    batch_size = pred_pose_shape.shape[0]
    sampled_idxs = np.round(np.random.sample(batch_size) * (len(mosh['pose']) - 2)).astype(np.int)
    mosh_pose = torch.tensor(deepcopy(mosh['pose'][sampled_idxs].astype(np.float32)))
    mosh_shape = torch.tensor(deepcopy(mosh['shape'][sampled_idxs].astype(np.float32)))
    mosh_pose_shape = torch.cat([batch_rodrigues(mosh_pose.view(-1, 3)).view(batch_size, -1), mosh_shape], dim=1)

    loss_disc, adv_loss_fake, adv_loss_real = adversarial_loss(discriminator, pred_pose_shape, mosh_pose_shape)
    losses.update({
        'loss_disc': loss_disc,
        'adv_loss_fake': adv_loss_fake,
        'adv_loss_real': adv_loss_real
    })
    for k, v in re_weight.items():
        losses[k] *= v

    tag_tail = mode
    loss, adv_loss, log_vars = parse_adv_losses(losses, tag_tail)

    # if loss.item() > 1:
    #     print('=' * 30, 'start of meta', '=' * 30)
    #     print([(i['idx'].item(), i['flip']) for i in data['img_meta'].data[0]])
    #     print('=' * 30, 'end of meta', '=' * 30)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data),
        adv_loss=adv_loss
    )

    if kwargs.get('log_grad', False):
        with torch.no_grad():
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm()
                    total_norm += param_norm.item()
            if total_norm:
                outputs['log_vars'][f'total_grad/{tag_tail}'] = total_norm

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        if issubclass(optimizer_cfg['type'], torch.optim.Optimizer):
            optimizer_cls = optimizer_cfg.pop('type')
        else:
            optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = getattr(datasets, val_dataset_cfg.type)
            if issubclass(dataset_type, datasets.CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    # import mmcv.runner.hooks.logger as mmcv_logger
    # mmcv_logger.LoggerHook
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # To crete a `latest.pth` file to run recursively
    # runner._epoch -= 1
    # runner.save_checkpoint(cfg.work_dir, filename_tmpl='dummy_{}.pth')
    # runner._epoch += 1
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_smpl_detector_fuse(model, datasets, cfg, **kwargs):
    # prepare data loaders
    data_loaders = [
        build_dataloader_fuse(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for dataset in datasets
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    # TODO: Build up a logger here that inherit the hook class
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    val_dataset_cfg = cfg.data.val
    eval_cfg = cfg.get('evaluation', {})
    # No need to add hook for validation.
    # We shall return the losses inside the loss function.
    # I won't re-use the code for the Evaluation Hook.
    # The evaluation result will be passed to log only.

    # To crete a `latest.pth` file to run recursively
    if kwargs.get('create_dummy', False):
        print('Create a dummy checkpoint for recursive training')
        runner._epoch -= 1
        runner.save_checkpoint(cfg.work_dir, filename_tmpl='dummy_{}.pth')
        runner._epoch += 1
        return

    pretrain_path = kwargs.get('load_pretrain', None)
    if kwargs.get('load_pretrain', None):
        print(f"Load pretrained model from {pretrain_path}")
        runner._epoch -= 1
        runner.load_checkpoint(pretrain_path)
        runner.save_checkpoint(cfg.work_dir, filename_tmpl='pretrained_{}.pth')
        runner._epoch += 1
        return

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs,
               time_limit=getattr(cfg, 'time_limit', None))


def train_adv_smpl_detector(model, datasets, cfg, **kwargs):
    # prepare data loaders
    data_loaders = [
        build_dataloader_fuse(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for dataset in datasets
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    adv_model = nn.DataParallel(Discriminator()).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    adv_optimizer = build_optimizer(adv_model, cfg.adv_optimizer)

    global runner
    runner = AdvRunner(adv_model, adv_optimizer, model, adv_batch_processor, optimizer, cfg.work_dir,
                       cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        assert NotImplementedError("AdvOptimizer is not implemented for fp16 yet")
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    # TODO: Build up a logger here that inherit the hook class
    # import mmcv.runner.hooks.logger as mmcv_logger
    # mmcv_logger.LoggerHook
    runner.register_training_hooks(cfg.adv_optimizer_config, cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    val_dataset_cfg = cfg.data.val
    eval_cfg = cfg.get('evaluation', {})
    # No need to add hook for validation.
    # We shall return the losses inside the loss function.
    # I won't re-use the code for the Evaluation Hook.
    # The evaluation result will be passed to log only.

    # To crete a `latest.pth` file to run recursively
    if kwargs.get('create_dummy', False):
        print('Create a dummy checkpoint for recursive training')
        runner._epoch -= 1
        runner.save_checkpoint(cfg.work_dir, filename_tmpl='dummy_{}.pth')
        runner._epoch += 1
        return

    pretrain_path = kwargs.get('load_pretrain', None)
    if kwargs.get('load_pretrain', None):
        print(f"Load pretrained model from {pretrain_path}")
        runner._epoch -= 1
        runner.load_checkpoint(pretrain_path)
        adv_pretrain_path = osp.join(*osp.split(pretrain_path)[:-1], 'adv_' + osp.split(pretrain_path)[-1])
        if osp.isfile(adv_pretrain_path):
            runner.load_adv_checkpoint(adv_pretrain_path)
        else:
            print('No adversarial checkpoint is found.')
        runner.save_checkpoint(cfg.work_dir, filename_tmpl='pretrained_{}.pth')
        runner._epoch += 1
        return

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    re_weight = cfg.re_weight if hasattr(cfg, 're_weight') else dict()
    mosh_path = cfg.common_train_cfg.mosh_path
    mosh_data = np.load(mosh_path)
    mosh = {'shape': mosh_data['shape'].copy(), 'pose': mosh_data['pose'].copy()}
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs,
               time_limit=getattr(cfg, 'time_limit', None), re_weight=re_weight, mosh=mosh,
               log_grad=cfg.get('log_grad', False))


# Change made for SLRUM preemption
import signal

runner = None


def safe_exit(signal_num, ec=0):
    import sys
    sys.stdout.flush()
    global runner
    runner.exit_code = ec


signal.signal(signal.SIGTERM, safe_exit)
