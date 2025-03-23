# ============================ #
# IMPORTS                      #
# ============================ #

import sys

sys.path.append("../../")
from api import CXRImage
from PIL import Image
import torch
import torchvision.transforms as transforms

from models.CarinaNet.CarinaNetModel import CarinaNetModel

# ============================ #
# LOAD MODELS                  #
# ============================ #

model_carinanet = CarinaNetModel(
    "checkpoints/finetuned_carinanet.pth", update_method=None
)

# ============================ #
# IMAGE PROCESSING             #
# ============================ #


def preprocess_image_carinanet(image_path, target_size=640):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.BICUBIC)
    transform = transforms.ToTensor()
    return transform(image)


# ============================ #
# INFERENCE FUNCTIONS          #
# ============================ #


def get_carinanet_predictions(cxr_image, resized_dim):
    tensor_image = preprocess_image_carinanet(cxr_image.image_path, resized_dim)
    if torch.cuda.is_available():
        tensor_image = tensor_image.cuda()
    images_and_ids = [(tensor_image, cxr_image.rid)]
    predictions = model_carinanet.predict(images_and_ids)
    # predictions = {cxr_image.rid: [(254.5, 182.0), (253.5, 124.5)]} # debug
    preds = {
        "carina": predictions[cxr_image.rid][0],
        "ett": predictions[cxr_image.rid][1],
    }
    return preds


def find_ett_carinanet(cxr_image: CXRImage, object_name: str):
    """Find the ETT location in an image."""
    resized_dim = 640
    preds = get_carinanet_predictions(cxr_image, resized_dim)
    point_prediction = preds["ett"]
    print(point_prediction)
    rescaled = cxr_image.rescale_point(
        point_prediction,
        (resized_dim, resized_dim),
    )
    return [cxr_image.point_to_bbox(rescaled)]


def find_carina_carinanet(cxr_image: CXRImage, object_name: str):
    """Find the Carina location in an image."""
    resized_dim = 640
    preds = get_carinanet_predictions(cxr_image, resized_dim)
    point_prediction = preds["carina"]
    rescaled = cxr_image.rescale_point(
        point_prediction,
        (resized_dim, resized_dim),
    )
    return [cxr_image.point_to_bbox(rescaled)]
