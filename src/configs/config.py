import os 
import numpy as np
import pandas as pd


DATA_ROOT = '../data'

PATHS = {
#     'TRAIN': 'train_full_size',
    'TRAIN': 'train_shrinked',
    'EXT': 'external_data',
    'TEST': 'test_shrinked',
    'CSV': 'csv',
    'XML': 'xmls'
}

for k, v in PATHS.items():
    PATHS[k] = os.path.join(DATA_ROOT, v)

PATHS['DATA'] = DATA_ROOT

PARAMS = {
    'PATHS': PATHS,
    'SEED': 42,
    'NB_INFERS': 3,
    'NB_FOLDS': 5,
    'SUPPORT_CLASS_AMOUNT': 400,
    'SUPPORT_POWER': .8,
    'BATCH_SIZE': 32,
    'VALID_BATCH_SIZE': 28,
    'DROPOUT': .5,
    'NB_EPOCHS': 50,
    'EPOCHS_PER_SAVE': 10,
    'NB_FREEZED_EPOCHS': 2,
    
    'SIMPLEX_NOISE': True,
    'SIDE': None,

    'LR': 5e-4,
    'MIN_LR': 1e-4,
    'EXP_GAMMA': .97,

    'CUDA_DEVICES': [0, 1],
    'LR_POLICE': [[0, 5e-4], [8, 1e-5], [12, 5e-6], [45, 1e-6]],
    'PLOT_KEYS': [
      'f1_score', 'loss'
    ],

    'SHRINKED_FULL_SIZE': 512,
    'SHRINKED_SIDE': 512,
    'USE_EXTERNAL_DATA': False,
    'USE_PSEUDOLABELS_DATA': True,
}

label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes", 
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}


label_names_ext = {
    28: "Vesicles",
    29: "Nucleus",
    30: "Midbody",
    31: "Cell Junctions",
    32: "Midbody ring",
    33: "Cleavage furrow",
}
all_label_names = label_names.copy()
all_label_names.update(label_names_ext)

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row

label_names_list = list(label_names.values())

labels_csv = os.path.join(PATHS['CSV'], 'prepared_train.csv')
external_csv = os.path.join(PATHS['CSV'], 'prepared_external.csv')
pseudo_csv = os.path.join(PATHS['CSV'], 'pseudolabels.csv')

# Dataset preparation
if not os.path.isfile(labels_csv):
    labels = pd.read_csv(os.path.join(PATHS['CSV'], 'train.csv'))
    for key in label_names.keys():
        labels[label_names[key]] = 0

    labels = labels.apply(fill_targets, axis=1)
    labels['Target'] = labels.Target.apply(lambda x: " ".join(map(str, x)))
    labels.to_csv(labels_csv, index=False)

labels = pd.read_csv(labels_csv)
labels['Type'] = 0

# Appearance of each class in dataset
appearance_train_only = dict(labels.drop(['Id', 'Target', 'Type'], axis=1).sum())

if PARAMS['USE_EXTERNAL_DATA'] and os.path.isfile(external_csv):
    elabels = pd.read_csv(external_csv)
    elabels['Type'] = 1
    labels = pd.concat([labels, elabels])
    labels.reset_index(drop=True, inplace=True)
#     labels.Target.apply(lambda x: np.array(x.split(" ")).astype(np.int))

if PARAMS['USE_PSEUDOLABELS_DATA'] and os.path.isfile(external_csv):
    plabels = pd.read_csv(pseudo_csv)
    plabels['Type'] = 2
    labels = pd.concat([labels, plabels])
    labels.reset_index(drop=True, inplace=True)

labels['Target'] = labels.Target.apply(lambda x: np.array(x.split(' ')).astype(np.int))

# Appearance of each class in dataset
appearance = dict(labels.drop(['Id', 'Target', 'Type'], axis=1).sum())
reverse_appearance = { reverse_train_labels[k]: v for k, v in appearance.items() }
