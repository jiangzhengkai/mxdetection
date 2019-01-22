import logging
import mxnet as mx
import math

class WarmupMultiFactorScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, step, factor=0.1, frequent=50, warmup=False, warmup_linear=True,
                 warmup_lr=0., warmup_end_lr=0., warmup_step=0):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_linear = warmup_linear
        self.warmup_lr = warmup_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_step = warmup_step
        self.frequent = frequent
        self.old_num_update = 0

    def __call__(self, num_update):
        if self.warmup and num_update <= self.warmup_step:
            if not self.warmup_linear:
                return self.warmup_lr
            else:
                self.base_lr = self.warmup_lr + (self.warmup_end_lr - self.warmup_lr) / self.warmup_step * num_update
                if num_update % self.frequent == 0 and num_update != self.old_num_update:
                    logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.base_lr)
                    self.old_num_update = num_update
                return self.base_lr
        while self.cur_step_ind <= len(self.step) - 1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr


class PolyScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, max_iter, power=0.9, frequent=50):
        super(PolyScheduler, self).__init__()
        self.max_iter = float(max_iter)
        self.power = power
        self.frequent = frequent
        self.old_num_update = 0
    
    def __call__(self, num_update):
        x = 1 - num_update / self.max_iter
        lr = self.base_lr * math.pow(x, self.power)
        if num_update % self.frequent == 0 and num_update != self.old_num_update:
            logging.info("iter = %d, lr = %f" % (num_update, lr))
            self.old_num_update = num_update
        return lr

def default_lr_scheduler(config):
    config.TRAIN.solver.epoch_size = config.TRAIN.num_examples // config.TRAIN.batch_size
    config.TRAIN.solver.begin_epoch = config.TRAIN.solver.load_epoch if config.TRAIN.solver.load_epoch else 0
    config.TRAIN.solver.begin_num_update = config.TRAIN.solver.epoch_size * config.TRAIN.solver.begin_epoch
    config.TRAIN.solver.max_iter = config.TRAIN.solver.epoch_size * config.TRAIN.solver.num_epoch
    logging.info('lr = %f, num_examples = %d, epoch_size = %d, max_iter = %d' %
                 (config.TRAIN.solver.lr, config.TRAIN.num_examples, config.TRAIN.solver.epoch_size, config.TRAIN.solver.max_iter))
    return config


def get_warmupmf_scheduler(config):
    config = default_lr_scheduler(config)
    step_epochs = [float(l) for l in config.TRAIN.solver.lr_step.split(',')]
    steps = [int(config.TRAIN.solver.epoch_size * x) for x in step_epochs]
    warmupmf_scheduler = WarmupMultiFactorScheduler(step=steps,
                                                    factor=0.1,
                                                    frequent=config.TRAIN.solver.frequent,
                                                    warmup=config.TRAIN.solver.warmup,
                                                    warmup_linear=config.TRAIN.solver.warmup_linear,
                                                    warmup_lr=config.TRAIN.solver.warmup_lr,
                                                    warmup_end_lr=config.TRAIN.solver.lr,
                                                    warmup_step=int(config.TRAIN.solver.epoch_size * config.TRAIN.solver.warmup_step_ratio))
    return config, warmupmf_scheduler


def get_poly_scheduler(config):
    config = default_lr_scheduler(config)
    poly_scheduler = PolyScheduler(max_iter=config.TRAIN.solver.max_iter,
                                   frequent=config.TRAIN.solver.frequent)
    return config, poly_scheduler
