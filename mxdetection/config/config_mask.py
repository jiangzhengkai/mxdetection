from easydict import EasyDict as edict


def add_mask_params(config):
    if 'TRAIN' not in config:
        config.TRAIN = edict()
    if 'TEST' not in config:
        config.TEST = edict()
    if 'network' not in config:
        config.network = edict()
    if 'dataset' not in config:
        config.dataset = edict()

    config.network.mask_pooled_size = (28, 28)

    config.TRAIN.mask_roi_batch_size = 4
    config.TRAIN.mask_loss_weights = [1.0, ]

    return config