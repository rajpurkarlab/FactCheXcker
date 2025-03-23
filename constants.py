from pathlib import Path

# Paths
MIMIC_CARINA_CACHE = "<Update Here!>"
MIMIC_ETT_CACHE = "<Update Here!>"
CHEXPERT_PLUS_CARINA_CACHE = "<Update Here!>"
CHEXPERT_PLUS_ETT_CACHE = "<Update Here!>"
SEGMENTATION_CACHE = "<Update Here!>"


# OpenAI Inference
OPENAI_API_VERSION = "<Update Here!>"
OPENAI_AZURE_ENDPOINT = "<Update Here!>"
OPENAI_MODEL_NAME = "<Update Here!>"

# MODULES
SEGMENTATION_TARGETS = [
    "Left Clavicle",
    "Right Clavicle",
    "Left Scapula",
    "Right Scapula",
    "Left Lung",
    "Right Lung",
    "Left Hilus Pulmonis",
    "Right Hilus Pulmonis",
    "Heart",
    "Aorta",
    "Facies Diaphragmatica",
    "Mediastinum",
    "Weasand",
    "Spine",
]

ENDOTRACHEAL_TUBE_SYNONYMS = ["endotracheal tube tip", "endotracheal tube", "ETT"]
