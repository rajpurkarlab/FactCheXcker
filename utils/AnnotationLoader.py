import os
from PIL import Image
import kwcoco
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils.constants import ANNO_FILE_NAME_FIELD, ANNO_IMAGE_ID_FIELD
from utils.utils import convert_coco_annot_to_tensors, get_annotations_by_image_id, get_image_file_path

class AnnotationLoader:
    def __init__(self, coco_annotations: kwcoco.CocoDataset):
        self.coco_annotations = coco_annotations

        self.image_meta = pd.DataFrame(list(coco_annotations.imgs.values()))

    def __add__(self, other):
        return AnnotationLoader(
            kwcoco.CocoDataset.union(self.coco_annotations, other.coco_annotations))
    
    def reset_image_meta(self, image_meta: pd.DataFrame):
        self.image_meta = image_meta
        new_gids = list(self.image_meta["id"])

        self.coco_annotations = self.coco_annotations.subset(new_gids)
    
    def get_all_image_meta(self):
        return self.image_meta
    
    def get_annotations_by_image_id(self, image_id: str) -> torch.Tensor:
        coco_annots = get_annotations_by_image_id(self.coco_annotations, image_id)
        return convert_coco_annot_to_tensors(coco_annots)
