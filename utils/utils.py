import os
import unicodedata

import kwcoco
import numpy as np
import pandas as pd
from utils.constants import *


def is_true(config, field):
    return (field in config) and (config[field])


def normalize_hospital_name(hospital_name):
    return unicodedata.normalize("NFC", hospital_name)


def get_annotation_file_path(
    data_dir: str,
    annotation_filename: str = None,
) -> str:
    if annotation_filename:
        return os.path.join(data_dir, DATA_ANNOTATION_DIR, annotation_filename)
    else:
        return os.path.join(data_dir, DATA_ANNOTATION_FILENAME)


def get_image_file_path(data_dir: str, data_source: str, file_name: str) -> str:
    return os.path.join(
        data_dir, DATA_IMAGE_DIR, data_source, file_name.replace(".jpg", ".png")
    )


def get_annotations_by_image_id(
    coco_annotations: kwcoco.CocoDataset, coco_image_id: int
) -> list[dict]:
    coco_anno_ids = coco_annotations.gid_to_aids[coco_image_id]
    coco_annos = [coco_annotations.anns[id] for id in coco_anno_ids]

    return coco_annos

def convert_coco_annot_to_tensors(coco_annots: list[dict]) -> np.ndarray:
    # get ground truth annotations
    annotations = np.zeros((0, 5))

    for coco_annot in coco_annots:
        annotation = np.zeros((1, 5))
        annotation[0, :4] = coco_annot["bbox"]
        annotation[0, 4] = COCO_LABELS_INVERSE[coco_annot["category_id"]]
        annotations = np.append(annotations, annotation, axis=0)

    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    return torch.tensor(annotations)

def get_image_filename_by_image_id(
    coco_annotations: kwcoco.CocoDataset, coco_image_id: int
) -> str:
    return coco_annotations.imgs[coco_image_id][ANNO_FILE_NAME_FIELD]


def get_center_coordinates_from_bbox(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x + w / 2, y + h / 2


def save_results(
    results: list[pd.DataFrame],
    output_path: str,
    cl_context: str,
    config: dict,
) -> None:
    results = pd.concat(results, ignore_index=True)
    ewc_lambda = (
        f"_lamb_{config['EWC_lambda']}" if config["update_method"] == EWC_UPDATE else ""
    )
    
    output_file_path = os.path.join(
        output_path,
        f'{config["model_type"]}_{cl_context}_{config[UPDATE_ORDER]}_{config[UPDATE_METHOD]}{ewc_lambda}.csv',
    )

    results.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")
