from mmcv.runner.hooks.optimizer import OptimizerHook


class AdvOptimizerHook(OptimizerHook):

    def __int__(self, **kwargs):
        super(AdvOptimizerHook, self).__init__(**kwargs)

    def after_train_iter(self, runner):
        runner.adv_optimizer.zero_grad()
        runner.outputs['adv_loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.adv_model.parameters())
        runner.adv_optimizer.step()
