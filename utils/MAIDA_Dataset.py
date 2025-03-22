import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils.constants import ANNO_FILE_NAME_FIELD, ANNO_IMAGE_ID_FIELD
from utils.utils import get_image_file_path


class MAIDA_Dataset(Dataset):

    def __init__(
        self,
        data_path: str = None,
        data_source: str = None,
        image_meta: pd.DataFrame = None,
        dataset: Dataset = None,
    ):
        """
        The images Dataset contains is an overset of that in image_meta. WE use image_meta to
        filter out the images we want in Dataset
        """
        if dataset is not None:
            self.image_ids = dataset.image_ids
            self.id_to_path = dataset.id_to_path
            self.image_meta = dataset.image_meta

        else:
            self.image_ids = image_meta.loc[
                ~image_meta[ANNO_FILE_NAME_FIELD].str.contains("crop"),
                ANNO_IMAGE_ID_FIELD,
            ].tolist()
            self.id_to_path = (
                image_meta.set_index(ANNO_IMAGE_ID_FIELD)[ANNO_FILE_NAME_FIELD]
                .apply(lambda f: get_image_file_path(data_path, data_source, f))
                .to_dict()
            )
            self.image_meta = image_meta

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        image_path = self.id_to_path[image_id]

        image = Image.open(image_path)
        transform = transforms.ToTensor()
        tensor_image = transform(image)

        # image_aug_path = image_path.split(".png")[0] + "_crop_top.png"
        if False:  # os.path.exists(image_aug_path):
            image_aug_id = self.path_aug_to_id[image_aug_path]
            image_aug = Image.open(image_aug_path)
            tensor_image_aug = transform(image_aug)
        else:
            image_aug_id = -1
            tensor_image_aug = torch.zeros_like(tensor_image)

        # Return the image as a tensor along with its index
        return {
            "image": tensor_image,
            "image_id": image_id,
            "image_aug": tensor_image_aug,
            "image_aug_id": image_aug_id,
        }

    def reset_image_meta(self, image_meta: pd.DataFrame) -> None:
        self.image_meta = image_meta
        return

    def get_image_meta(self) -> pd.DataFrame:
        return self.image_meta

    def get_image_by_image_id(self, image_id: int) -> torch.Tensor:
        image_path = self.id_to_path[image_id]
        image = Image.open(image_path)
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        return tensor_image
