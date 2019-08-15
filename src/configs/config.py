from pathlib import Path
import os
import easydict

CROP_SIDE = 256
SIDE = 130
MEAN = [0.15817458]
STD = [0.16162706]
NUM_CLASSES = 9
NUM_EPOCHS = 200
TRAIN_TEST_SPLIT = .85
SEED = 42
BBOXES_SIDE = 32

PARAMS = easydict.EasyDict()
PARAMS.LR = 1e-4
PARAMS.BATCH_SIZE = 1
PARAMS.EXP_GAMMA = .9
PARAMS.THRESHOLD = .5
PARAMS.CUDA_DEVICES = [0]


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