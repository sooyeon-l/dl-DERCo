from pathlib import Path

SAVE_PATH = Path('/workspace/data')
INPUT_PATH = SAVE_PATH / 'inputs'
CHECKPOINTS_PATH = SAVE_PATH / 'checkpoints'
ANALYSIS_SAVE_PATH = SAVE_PATH / 'analysis'
RUN_OUTPUTS_PATH = SAVE_PATH / 'runs'
CONFIG_SNAPS_PATH = SAVE_PATH / 'config_snapshots'

LOG_EVERY = 5

# Move these later
# CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
# OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

ORIG_SFREQ = 1000
TARGET_SFREQ = 250
DOWNSAMPLE_UP = 1
DOWNSAMPLE_DOWN = 4

N_TIMEPOINTS = {
    1000: 801,
    250: 201,
}

WINDOWS = {
    1000: {
        '0800':   slice(0, 801),
        '0200':   slice(0, 201),
        '300500': slice(300, 501),
        '500800': slice(500, 801),
    },
    250: {
        '0800':   slice(0, 201),
        '0200':   slice(0, 51),
        '300500': slice(75, 126),
        '500800': slice(125, 201),
    },
}

EEGNET_WINDOWS = ['0800']
CNN_WINDOWS    = ['0800', '0200', '300500', '500800']

NUM_CHANNELS = 32

CNN_CONFIGS = {
    1000: {
        "kernel_len": 125,
        "padding": 62,
    },
    250: {
        "kernel_len": 31,
        "padding": 15,
    },
}

EEGNET_CONFIGS = {
    250: {
        "F1": 8,
        "D": 2,
        "F2": 16,
        "kernel_len": 125,
        "kernel_padding": 62,
        "sep_kernel_len": 31,
        "sep_padding": 15,
        "dropout": 0.5,
    }
}

BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 1e-3
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5 # Because early stopping patience is 15 - gives scheduler time to reduce LR before training stops
DROPOUT_P = {
    'classifier': 0.5, 
    'conv': 0.25, 
    'eegnet': 0.5
}
GRAD_CLIP = 1.0
RANDOM_SEED = 42
WEIGHT_DECAY = 0.01
N_FOLDS = 5
THRESHOLD = 0.5