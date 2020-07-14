from __future__ import division

import argparse
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)

import torch

torch.multiprocessing.set_sharing_strategy('file_system')

print(sys.executable)
from mmcv import Config
from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed, train_smpl_detector_fuse, train_adv_smpl_detector)
from mmdet.models import build_detector
from mmdet.datasets.concat_dataset import ConcatDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--create_dummy', action='store_true',
                        help='Create a dummy checkpoint for recursive training on clusters')
    parser.add_argument('--load_pretrain', type=str, default=None,
                        help='Load parameters pretrained model and save it for recursive training')
    parser.add_argument('--imgs_per_gpu', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.imgs_per_gpu > 0:
        cfg.data.imgs_per_gpu = args.imgs_per_gpu

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if hasattr(cfg, 'fuse') and cfg.fuse:
        train_dataset = get_dataset(cfg.datasets[0].train)
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=train_dataset.CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = train_dataset.CLASSES
        datasets = list()
        for flow in cfg.workflow:
            mode, epoches = flow
            cur_datasets = list()
            for dataset_cfg in cfg.datasets:
                if hasattr(dataset_cfg, mode):
                    cur_datasets.append(get_dataset(getattr(dataset_cfg, mode)))
            datasets.append(ConcatDataset(cur_datasets))
        val_dataset = None
        if cfg.data.train.get('val_every', None):
            val_dataset = list()
            for dataset_cfg in cfg.datasets:
                if hasattr(dataset_cfg, 'val'):
                    val_dataset.append(get_dataset(dataset_cfg.val))
            val_dataset = ConcatDataset(val_dataset)
        if hasattr(cfg.model, 'smpl_head') and cfg.model.smpl_head.loss_cfg.get('adversarial_cfg', False):
            train_adv_smpl_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=args.validate,
                logger=logger,
                create_dummy=args.create_dummy,
                val_dataset=val_dataset,
                load_pretrain=args.load_pretrain,
            )
        else:
            train_smpl_detector_fuse(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=args.validate,
                logger=logger,
                create_dummy=args.create_dummy,
                val_dataset=val_dataset,
                load_pretrain=args.load_pretrain
            )


if __name__ == '__main__':
    main()
