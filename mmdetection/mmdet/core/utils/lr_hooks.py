from mmcv.runner.hooks.lr_updater import LrUpdaterHook


class PowerLrUpdaterHook(LrUpdaterHook):

    def __init__(self, step, gamma=1., **kwargs):
        assert isinstance(step, int) and step > 0
        self.step = step
        self.gamma = gamma
        super(PowerLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        return base_lr if progress == 0 or progress % self.step != 0 else base_lr * self.gamma


class SequenceLrUpdaterHook(LrUpdaterHook):

    def __init__(self, seq, **kwargs):
        assert isinstance(seq, list) and len(seq) > 0
        self.seq = seq
        super(SequenceLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        return self.seq[progress] if progress < len(self.seq) else self.seq[-1]
