import random
import torch

CUDA_AVAILABLE = True # CUDA_AVAILABLE
print(f"CUDA available: {CUDA_AVAILABLE}")

SEED = random.randint(0, 100000) 
WANDB_OFF = False
FINE_TUNING = "fine_tune"
HYPERPARAMETER_TUNING = "hyperparameter_tuning"


# Expected directory and file names
CONFIG_DIR = "configs"
INFERENCE_DIR = "inference"
FINE_TUNE_DIR = "fine_tune"
HYPER_TUNE_DIR = "hyperparameter_tuning"
CL_DIR = "CL"
SWEEP_CONFIG_FILENAME = "sweep_config.yaml"

#### Models
DEFAULT_MODEL_NAME = "model.pt"
CARINA_NET_OTS_MODEL_DIR = "models/CarinaNet"  # default model directory for CarinaNet
USE_CARINA_NET_HOSPITAL_FINETUNED = "hospital-specific"  # TODO remove
OFF_THE_SHELF = "OTS"
ALL_BUT_TARGET_HOSPITALS_ONLY_FINETUNED = "all-but-garget-hospitals-only"
TARGET_HOSPITAL_ONLY_FINETUNED = "target-hospital-only"
PUBLIC_ONLY_FINETUNED = "public-only"
PUBLIC_HOSPITALS_FINETUNED = "public-hospitals"
TARGET_HOSPITAL_FINETUNED = "target-hospital"

### Metrics
CLASSIFICATION_LOSS = "classification_loss"
REGRESSION_LOSS = "regression_loss"
TOTAL_LOSS = "total_loss"
ERROR_SUFFIX = "-error"
RECALL_SUFFIX = "-recall"


#### Inference
OTS_MODEL_INFERENCE = "OTS"
INFERENCE_DATASET = "inference-dataset"
USE_PUBLIC_DATASET_FOR_INFERENCE = "public"
USE_HOSPITLAS_DATASET_FOR_INFERENCE = "hospitals"
USE_TARGET_HOSPITAL_DATASET_FOR_INFERENCE = "target-hospital"
INFERENCE_OUTPUT_FILENAME = "inference.csv"

### Finetune
USE_ALL_BUT_TARGET_HOSPITALS_ONLY = "use_all-but-target-hospital_only"
FINETUNE_OUTPUT_FILENAME = "finetuned"

#### Continual Learning
INTRA_HOSPITAL_CL = "intra"
GLOBAL_CL = "global"
GLOBAL_HOSPITAL_MIX = "hospital-mix"
GLOBAL_SEQUENTIAL = "sequential"
NAIVE_UPDATE = "naive"
EWC_UPDATE = "EWC"
REHEARSAL_UPDATE = "rehearsal"

### Fields for update_dict
SIMULATION_IDX = "simulation_idx"
PATIENT_IDX_INIT = "patient_idx_init"
ITERATION_IDX_INIT = "iteration_idx_init"
BATCH_IDX = "batch_idx"
BATCH_SIZE = "batch_size"
PREV_DATA = "prev_data"
CURR_DOMAIN_NAME = "curr_domain_name"
CURR_DOMAIN_DATA = "curr_domain_data"
PREV_DOMAIN_NAME = "prev_domain_name"
IS_FINAL_UPDATE = "is_final_update"

### Fields in config
UPDATE_ORDER = "update_order" # choose among sequential, hospital mix, and full updates
MODEL_TYPE = "model_type" # choose among OTS, public-only, public-hospitals, target-hospital, and hospital-specific models
UPDATE_METHOD = "update_method" # choose among EWC, naive, and rehearsal
MODEL_PATH = "model_path"
DATA_PATH = "data_path"
OUTPUT_PATH = "output_path"
TARGET_HOSPITAL = "target_hospital"
BATCH_SIZE = "batch_size"
NUM_SIM = "number_of_simulation"
IS_HYPER_TUNING = "is_hyperparameter_tuning"
LEARNING_RATE = "learning_rate"
WEIGHT_DECAY_FIELD = "weight_decay"
T0_FIELD = "T_0"
HAS_L2INIT = "has_L2init"

WANDB_PROJECT_NAME = "wandb_project_name"


### Fields in the prediction dataframe
ITERATION_FIELD = "iteration"
HOSPITAL_NAME_FIELD = "hospital_name"
SIMULATION_FIELD = "simulation"
HOSPITAL_ORDER_FIELD = "hospital_order"
INDEX_HOSPITAL_FIELD = "index_hospital"

### We only support these data sources
HOSPITAL_DATA_SOURCE = "hospitals"
TEST_DATA_SOURCE = "test"
TRAIN_DATA_SOURCE = "train"
VAL_DATA_SOURCE = "val"

### Data
DATA_IMAGE_DIR = "images"
DATA_ANNOTATION_DIR = "annotations"
DATA_ANNOTATION_FILENAME = "annotations.json"

### Fields in the annotation file
ANNO_IMAGES_FIELD = "images"
ANNO_HOSPITAL_NAME_FIELD = "hospital_name"
ANNO_FILE_NAME_FIELD = "file_name"
ANNO_IMAGE_ID_FIELD = "id"
ANNO_CAT_TIP = "tip"
ANNO_CAT_CARINA = "carina"

### Hyperparameters for CarinaNet
LEARNING_RATE = 0.00005 #0.00004628194
WEIGHT_DECAY = 0.04146686
MAX_LR = 0.00003433392
PCT_START = 0.2736
TOTAL_STEP = 100
T_0 = 10
EWC_LAMBDA = 1e6
NUM_EPOCHS = 10
L2_INIT_LAMBDA = 0.05
EARLY_STOPPING_PATIENCE = 1
WORKER_NUM = 2

### COCO labels conversion

# To be consistent with CarinaNet implementation
# Reference
# "categories": [
    # {"id": 1, "name": "carina", "raw_id": 3046},
    # {"id": 2, "name": "tip", "raw_id": 3047}
# ]
CAT_ID_CARINA = 1
CAT_ID_TIP = 2
COCO_LABELS = {0:CAT_ID_CARINA, 1:CAT_ID_TIP}
COCO_LABELS_INVERSE = {CAT_ID_CARINA:0, CAT_ID_TIP:1}

### Others
ANNOS_LOADER_KEY = "annos_loader"
DATA_LOADERS_KEY = "data_loaders"
SUFFIX_KEY = "suffix"
SKIP_SIMULATION = "skip_simulation"
ALL_KEY = "all"
