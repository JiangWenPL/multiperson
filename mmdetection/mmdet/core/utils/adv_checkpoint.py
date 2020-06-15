from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.utils import master_only


class AdvCheckpointHook(CheckpointHook):
    def __int__(self, **kwargs):
        super(AdvCheckpointHook, self).__init__(**kwargs)

    @master_only
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
