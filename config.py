import os
from easydict import EasyDict as edict

# sys config
config = edict()
config.count_flops = True
config.num_classes = 10576


# default settings
default = edict()
default.network = 'mobilefacenet'
# default dataset
default.dataset = 'webface'
default.image_shape = (112,112,3)
default.loss = 'arcface'
default.models_root = './checkpoints'
# optimizer params
default.lr = 0.1
default.lr_steps = 80000
default.wd = 0.0005
default.verbose = 2000
default.mom = 0.9
default.batch_size = 128
default.frequent = 20


default.end_epoch = 10000
default.ckpt = 3