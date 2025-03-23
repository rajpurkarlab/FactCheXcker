# ============================ #
# IMPORTS                      #
# ============================ #

import sys

sys.path.append("../")
from api import CXRImage
from models.LitResnet import LitResnet
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

# ============================ #
# LOAD MODELS                  #
# ============================ #

resnet_model = LitResnet.load_from_checkpoint("checkpoints/ett-resnet-30epoch.ckpt")
resnet_model.eval()


# ============================ #
# IMAGE PROCESSING             #
# ============================ #


def preprocess_image_resnet(image_path, target_size=512):
    # Open image with PIL
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),  # Resize to 512x512
            transforms.ToTensor(),  # Convert image to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Mean and standard deviation of the ImageNet images
        ]
    )
    return transform(image)


# ============================ #
# INFERENCE FUNCTIONS          #
# ============================ #


def exists_ett_resnet(cxr_image: CXRImage, object_name: str):
    """Check if ETT exists ResNet."""
    tensor_image = preprocess_image_resnet(cxr_image.image_path)
    if torch.cuda.is_available():
        tensor_image = tensor_image.cuda()
    tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension

    predictions = resnet_model(tensor_image)
    predictions = predictions.cpu().detach().numpy()  # send to cpu
    predicted_class_index = np.argmax(predictions)
    return bool(predicted_class_index)
