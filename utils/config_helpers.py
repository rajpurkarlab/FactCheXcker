import os
from utils.constants import *
from utils.utils import is_true

##### inference


def get_model_path(config):
    # If model_path is specified, use it
    if "model_path" in config and (config["model_path"] != ""):
        return config["model_path"]

    model_type = config["model_type"]

    if model_type == OFF_THE_SHELF:
        return os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME)

    elif model_type == PUBLIC_ONLY_FINETUNED:
        return os.path.join(
            config["output_path"],
            FINE_TUNE_DIR,
            PUBLIC_ONLY_FINETUNED,
            DEFAULT_MODEL_NAME,
        )

    elif (
        model_type == ALL_BUT_TARGET_HOSPITALS_ONLY_FINETUNED
        or model_type == PUBLIC_HOSPITALS_FINETUNED
    ):
        target_hospital_wo_space = config["target_hospital"].replace(" ", "_")
        return os.path.join(
            config["output_path"],
            FINE_TUNE_DIR,
            target_hospital_wo_space,
            model_type,
            DEFAULT_MODEL_NAME,
        )

    else:
        raise ValueError(f"Invalid model type: {model_type}")


def get_output_path_for_inference(config):
    return os.path.join(config["output_path"], INFERENCE_DIR, config["model_type"])

def get_output_path_for_global_CL(config):
    return os.path.join(config["output_path"], GLOBAL_CL, config[UPDATE_METHOD], config[SUFFIX_KEY])

def get_output_path_for_intra_hospital_CL(config):
    return os.path.join(config["output_path"], INTRA_HOSPITAL_CL, config[UPDATE_METHOD], config[SUFFIX_KEY])