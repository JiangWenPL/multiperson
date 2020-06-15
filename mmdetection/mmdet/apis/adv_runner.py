import mmcv
from mmcv.runner import Runner
import torch
from ..core.utils.adv_optimizer import AdvOptimizerHook
import os.path as osp
from mmcv.runner.checkpoint import save_checkpoint, load_checkpoint
from ..models.smpl_heads.smpl_common import Discriminator
import logging


class AdvRunner(Runner):

    def __init__(self, adv_model, adv_optimizer, *args, **kwargs):
        self.adv_optimizer = adv_optimizer
        self.adv_model = adv_model
        super(AdvRunner, self).__init__(*args, **kwargs)

    def register_training_hooks(self, adv_optimizer_config, *args, **kwargs):
        super(AdvRunner, self).register_training_hooks(*args, **kwargs)
        self.register_hook(self.build_hook(adv_optimizer_config, AdvOptimizerHook), priority='HIGH')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):

        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

        filename_tmpl = 'adv_' + filename_tmpl
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'adv_latest.pth')
        optimizer = self.adv_optimizer if save_optimizer else None
        save_checkpoint(self.adv_model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

    def load_adv_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.adv_model, filename, map_location, strict,
                               self.logger)

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        adv_checkpoint = checkpoint.replace('latest.pth', 'adv_latest.pth')
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
            adv_checkpoint = self.load_adv_checkpoint(
                adv_checkpoint, map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)
            adv_checkpoint = self.load_adv_checkpoint(
                adv_checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'optimizer' in adv_checkpoint and resume_optimizer:
            self.adv_optimizer.load_state_dict(adv_checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        super(AdvRunner, self).run(data_loaders, workflow, max_epochs, discriminator=self.adv_model, **kwargs)
