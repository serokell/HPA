from pathlib import Path
import os
import easydict

CROP_SIZE = 176
UPSCALE_FACTOR = 2

MEAN = 0.0361812449991703
STD = 0.0334203876554966

CROP_SIDE = 256
SIDE = 130
MEAN = [0.15817458]
STD = [0.16162706]
NUM_CLASSES = 9
NUM_EPOCHS = 200
TRAIN_TEST_SPLIT = .85
SEED = 42
BBOXES_SIDE = 32

COLORISATION = False

PARAMS = easydict.EasyDict()
PARAMS.LR = 1e-4
PARAMS.BATCH_SIZE = 20
PARAMS.MAX_STEPS_PER_EPOCH = -1
PARAMS.GCLIP = 1.
PARAMS.WD = 1e-4
PARAMS.THRESHOLD = .5
PARAMS.NUM_EPOCHS = 100
PARAMS.CUDA_DEVICES = [0]

PARAMS.COLORISATION = {
    "in_channels": 1,
    "out_channels": 1
}


PATHS = easydict.EasyDict()
PATHS.DATASET_ROOT = Path('/root/projects/HPA/data')

PATHS.DATA = Path('/root/projects/HPA/data')
PATHS.CSV = PATHS.DATA/'csv/'
PATHS.TRAIN = PATHS.DATA/'train/'
PATHS.VALID = PATHS.DATA/'valid/'
PATHS.TEST = PATHS.DATA/'test/'
PATHS.MODELS = PATHS.DATA/'models/'
PATHS = easydict.EasyDict({ k: v.resolve() for k, v in PATHS.items() })

PATHS.URI_LABEL_PREFIX = [
    'https://label.dev.dsai.io/storage/app/uploads/public/',
    'https://label.cmai.tech/storage/app/uploads/public/'
]
PATHS.PREPARED_PREFIX = 'prepared_'

def merge_opts(opt1, opt2):
    return easydict.EasyDict({**opt1, **opt2})

base_pix2pix_options = easydict.EasyDict({
    # experiment params
    'name': 'experiment_name',                  # name of the experiment
    'gpu_ids': [0],                             # used gpu ids
    'checkpoints_dir': '../pix2pix_training',   # where to save the models
    'epoch': 'latest',                          # model from which epoch to load
    'load_iter': 0,                             # model from each iteration to load (overrides epoch setting)
    'verbose': False,                           # is the model verbose
    'suffix': '',                               # optional suffix
    'direction': 'AtoB',                        # stupid option
    # model params
    'input_nc': 1,                              # number of input channels
    'output_nc': 3,                             # number of output channels
    'ngf': 64,                                  # number of generator filters in the last conv layer
    'ndf': 64,                                  # number of discriminator filters in the first conv layer
    'netD': 'basic',                            # discriminator architecture [basic | n_layers | pixel]
    'netG': 'resnet_9blocks',                   # generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
    'n_layers_D': 3,                            # only used if netD == n_layers
    'norm': 'instance',                         # instance normalization or batch normalization
    'init_type': 'normal',                      # network initialization [normal | xavier | kaiming | orthogonal]
    'init_gain': 0.02,                          # scaling factor for normal, xavier and orthogonal
    'no_dropout': False,                        # don't use dropout
    # dataloader params
    'dataset_type': 'hpa',                      # kind of dataset to prepare
    'num_workers': 8,                           # numbers of dataloader workers
    'batch_size': 20,                           # batch size
    'crop_size': 176,                           # crop images to this size
})

train_pix2pix_options = merge_opts(easydict.EasyDict({
    'print_freq': 100,                          # how many samples to process between prints
    'save_latest_freq': 5000,                   # frequency of saving latest results
    'save_epoch_freq': 1,                       # frequency of saving checkpoints at the end of epochs
    'save_by_iter': False,                      # whether saves model by iteration
    'continue_train': False,                    # continue training: load the latest model
    'val_results_subfolder': 'results',         # subfolder with model image results
    'epoch_count': 1,                           # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
    'phase': 'train',                           # train, val, test, etc
    'niter': 100,                               # # of iter at starting learning rate
    'niter_decay': 100,                         # # of iter to linearly decay learning rate to zero
    'beta1': 0.5,                               # momentum term of adam
    'lr': 0.0002,                               # initial learning rate for adam
    'lambda_L1': 100.0,                         # weight for L1 loss
    'gan_mode': 'lsgan',                        # the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
    'pool_size': 50,                            # the size of image buffer that stores previously generated images
    'lr_policy': 'linear',                      # learning rate policy. [linear | step | plateau | cosine]
    'lr_decay_iters': 50,                       # multiply by a gamma every lr_decay_iters iterations
}), base_pix2pix_options)