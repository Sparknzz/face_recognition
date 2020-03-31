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
default.models_root = '/root/face_recognition/checkpoints'
# optimizer params
default.lr = 0.1
default.lr_steps = 80000
default.wd = 0.0005
default.verbose = 2000
default.mom = 0.9
default.batch_size = 200
default.frequent = 20


default.end_epoch = 200
default.ckpt = 3




# new version 
import torch
configurations = {
    1: dict(
        SEED = 2020, # random seed for reproduce results

        DATA_ROOT = '/home/data/CASIA/CASIA-WebFace-Mask-Aligned', # the parent root where your train/val/test data are stored
        MODEL_ROOT = './checkpoints/mask_checkpoints', # the root to buffer your checkpoints
        LOG_ROOT = './logs', # the root to log your train/val status
        EVAL_ROOT = '/root/face_recognition/data/eval',

        BACKBONE_RESUME_ROOT = './checkpoints/mask_checkpoints/Backbone.pth', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './checkpoints/mask_checkpoints/Head_ArcFace.pth', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 256,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.01, # initial LR
        NUM_EPOCH = 100, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        # STAGES = [35, 65, 95], # epoch stages to decay learning rate
        STAGES = [15, 35, 60],

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
),
}

