from helpers import *
from constants import *

from typing import List, Tuple, Optional
import h5py
import torch
import math
import re


class ModuleCache:
    """A cache for storing precomputed segmentation and prediction data for chest X-rays.

    Attributes
    ----------
    raw_segmentations : Dict[str, torch.Tensor]
        Raw segmentation maps for each report ID (rid).
    bin_segmentations : Dict[str, torch.Tensor]
        Binary segmentation maps for each rid with shape [1, 14, 512, 512], where each
        channel corresponds to a specific segmentation target.
    carina_ett_preds : Dict
        Precomputed predictions for carina and endotracheal tube tip points.
    ett_exists_preds : Dict
        Precomputed presence predictions for endotracheal tube.
    segmentation_targets : List[str]
        List of segmentation target names, corresponding to the channels in bin_segmentations.
    """

    def __init__(self, dataset="MIMIC") -> None:
        if dataset == "MIMIC":
            self.carina_ett_preds = load_json(MIMIC_CARINA_CACHE)
            self.ett_exists_preds = load_json(MIMIC_ETT_CACHE)
        if dataset == "CHEXPERT-PLUS":
            self.carina_ett_preds = load_json(CHEXPERT_PLUS_CARINA_CACHE)
            self.ett_exists_preds = load_json(CHEXPERT_PLUS_ETT_CACHE)

        self.segmentation_targets = [
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
        self.segmentation_targets_lower = [
            target.lower() for target in self.segmentation_targets
        ]

    def point_to_bbox(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """Convert a point to a bounding box with zero area (centered on the point)."""
        x, y = point
        return (x, y, x, y)

    def rescale_x(self, value: float, original_width: int, model_width: int) -> float:
        """Rescale predicted image value"""
        return value * (original_width / model_width)

    def rescale_y(self, value: float, original_height: int, model_height: int) -> float:
        """Rescale predicted image value"""
        return value * (original_height / model_height)

    def rescale_point(
        self,
        point: Tuple[float, float],
        d_original: Tuple[float, float],
        d_model: Tuple[float, float],
    ) -> float:
        """Rescale predicted image value"""
        rescaled = (
            self.rescale_x(point[0], d_original[0], d_model[0]),
            self.rescale_y(point[1], d_original[1], d_model[1]),
        )
        return rescaled

    def get_object(
        self, rid: str, object_name: str, original_width: int, original_height: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """Retrieve the bounding boxes for a specified object in a given report ID.

        Parameters
        ----------
        rid : str
            Report ID to search within.
        object_name : str
            Name of the object to locate. Supported objects: "carina", "endotracheal tube".
        original_width: int
            Original image width
        original_height: int
            Original image height

        Returns
        -------
        Optional[List[[Tuple[float, float, float, float]]]
            A list of tuples representing the bounding box for the specified object,
            or None if the object or report ID is not found.

        supported = ["carina", "endotracheal tube"]
        """
        if object_name == "carina":  # assume only one carina
            if rid not in self.carina_ett_preds:
                return None
            print("Point Prediction for RID:", rid)
            point_prediction = self.carina_ett_preds[rid]["predictions"]["0"]["pred"]
            rescaled = self.rescale_point(
                point_prediction, (original_width, original_height), (640, 640)
            )
            return [self.point_to_bbox(rescaled)]

        if object_name in ENDOTRACHEAL_TUBE_SYNONYMS:  # assume only one ETT
            if rid not in self.carina_ett_preds:
                return None
            point_prediction = self.carina_ett_preds[rid]["predictions"]["1"]["pred"]
            rescaled = self.rescale_point(
                point_prediction, (original_width, original_height), (640, 640)
            )
            return [self.point_to_bbox(rescaled)]

        # Return None if the object_name is not supported.
        return None

    def check_exists(self, rid: str, object_name: str) -> Optional[bool]:
        """Check if a specified object exists in a given report ID.

        Parameters
        ----------
        rid : str
            Report ID to search within.
        object_name : str
            Name of the object to check for presence. Supported objects: "endotracheal tube".

        Returns
        -------
        Optional[bool]
            True if the specified object is present, False if absent, or None if data is unavailable.

        supported = ["carina", "endotracheal tube"]
        """
        if object_name in ENDOTRACHEAL_TUBE_SYNONYMS:
            prediction = self.ett_exists_preds[rid]["predictions"].get(
                "prediction", None
            )
            return prediction == "ET_present"
        if object_name == "carina":
            return True
        # Return None if the object_name is not supported or if the object is false
        return None

    def get_segmentation(self, rid: str, target: str) -> Optional[List[int]]:
        """Retrieve the segmentation map for a specific target in the report ID.

        Parameters
        ----------
        rid : str
            Report ID to retrieve the segmentation for.
        target : str
            Name of the segmentation target (e.g., "left lung").

        Returns
        -------
        Optional[List[int]]
            A 2D list of shape [512, 512] for the target segmentation, or None if not found.
        """
        try:
            with h5py.File(SEGMENTATION_CACHE, "r") as h5file:
                # Access segmentation maps for a specific key
                bin_segmentations = h5file[rid][:]
                print(bin_segmentations)

                # Get the index of the target and retrieve the corresponding channel
                target_idx = self.segmentation_targets_lower.index(target)
                segmentation_map = bin_segmentations[
                    target_idx
                ]  # Extract the target channel
                return segmentation_map
        except Exception as e:
            print(e)
            return None


class CXRObject:
    """A Python class containing an image object with bounding box information.

    Parameters
    ----------
    object_name : str
        The object name.
    bbox : Tuple[float, float, float, float]
        A tuple representing the bounding box in the format (left, lower, right, upper).

    The object is considered a point if left == right and lower == upper.
    """

    def __init__(
        self, object_name: str, bbox: Tuple[float, float, float, float]
    ) -> None:
        self.object_name = object_name
        self.bbox = bbox

    def is_point(self) -> bool:
        """Check if the bounding box is a single point."""
        left, lower, right, upper = self.bbox
        return left == right and lower == upper

    def get_center(self) -> Tuple[float, float]:
        """Get the center of the bounding box or the point if it's a single point."""
        if self.is_point():
            return (self.bbox[0], self.bbox[1])  # Return the exact point
        else:
            # Calculate the center of the bounding box
            left, lower, right, upper = self.bbox
            center = (
                (left + right) / 2,
                (lower + upper) / 2,
            )
            return center


class CXRSegmentation:
    def __init__(self, region_name, segmentation_map):
        self.region_name = region_name
        self.segmentation_map = segmentation_map

    def get_pixel_width(self):
        width = 0
        flattened = (self.segmentation_map == 1).any(dim=0).int()
        indices = torch.nonzero(flattened == 1).flatten()
        if len(indices) > 0:
            width = indices[-1].item() - indices[0].item()
        return width

    def get_pixel_height(self):
        height = 0
        flattened = (self.segmentation_map == 1).any(dim=1).int()
        indices = torch.nonzero(flattened == 1).flatten()
        if len(indices) > 0:
            height = indices[-1].item() - indices[0].item()
        return height


class CXRImage:
    """A Python class containing an image related to a report as well as relevant information.

    Parameters
    ----------
    rid: str
        Id of report object
    reports: dict
        Reports of "GroundTruth" and models.
    image_path: str
        Full image path.
    original_size: List[int]
        The original WxH of the image.
    pixel_spacing: Tuple[float, float]
        Pixel spacing (mm) in the x- and y-directions.
    cache: ModuleCache
        An instance of ModuleCache to retrieve precomputed segmentation and measurement data.

    Methods
    -------
    exists(object_name: str) -> bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    find(object_name: str) -> List[CXRObject]
        Returns a list of CXRObjects centered around any objects found in the image matching the object_name.
    segment(region_name: str) -> CXRSegmentation
        Returns a segmentation map of the image based on the region specified.
    within(ob: CXRObject, region: CXRSegmentation) -> bool
        Returns true of if the object center is within the region.
    distance(obj_a: CXRObject  obj_b: CXRObject) -> float
        Returns the distance (in cm) between the center of two objects in the image.
    diameter(obj: CXRObject) -> float
        Returns the diameter (in cm) of the CXRObject.
    dimensions(obj: CXRObject) -> Tuple[float, float]
        Returns the dimensions (in cm) of the CXRObject according to major axis x minor axis.
    width(segmentation: CXRSegmentation) -> float
        Returns the greatest width (in cm) of a segmentation.
    height(segmentation: CXRSegmentation) -> float
        Returns the greatest height (in cm) of a segmentation.
    filter(objects: [CXROBject], region: CXRSegmentation) -> List[CXRObject]
        Returns all objects with their center within the region.
    """

    def __init__(
        self,
        rid: str,
        image_path: str,
        reports: dict,
        original_size: List[int],
        pixel_spacing: Tuple[float, float],
        cache: ModuleCache,
    ):
        self.rid = rid
        self.image_path = image_path
        self.reports = reports
        self.original_width, self.original_height = original_size
        self.pixel_spacing = pixel_spacing
        self.cache = cache

    def normalize_query(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r"\s+", " ", query)
        query = query.strip()
        return query

    def rescale_x(self, value: float, width: int) -> float:
        """Rescale predicted image value"""
        return value * (self.original_width / width)

    def rescale_y(self, value: float, height: int) -> float:
        """Rescale predicted image value"""
        return value * (self.original_height / height)

    def convert_pixel_dx_to_cm(self, pixel_dx: float) -> float:
        """Convert a pixel distance in the x-direction to cm."""
        dx_cm = pixel_dx * self.pixel_spacing[0] / 10  # mm to cm
        return dx_cm

    def convert_pixel_dy_to_cm(self, pixel_dy: float) -> float:
        """Convert a pixel distance in the y-direction to cm."""
        dy_cm = pixel_dy * self.pixel_spacing[1] / 10  # mm to cm
        return dy_cm

    def distance(self, obj_a: CXRObject, obj_b: CXRObject) -> float:
        """Returns the distance (in cm) between two objects."""
        # get the middle of the bounding boxes
        center_a = obj_a.get_center()
        center_b = obj_b.get_center()

        # get cm distance
        p_dx = center_a[0] - center_b[0]
        p_dy = center_a[1] - center_b[1]
        cm_dx = self.convert_pixel_dx_to_cm(p_dx)
        cm_dy = self.convert_pixel_dy_to_cm(p_dy)
        cm_distance = math.sqrt(cm_dx**2 + cm_dy**2)

        return cm_distance

    def diameter(self, obj: CXRObject) -> float:
        """Returns the diameter (in cm) of the largest CXRObject."""

        # define point to have a diameter of 1mm
        if obj.is_point():
            return 0.1

        # diameter is right - left or lower - upper
        p_dx = abs(obj.bbox[2] - obj.bbox[0])
        cm_dx = self.convert_pixel_dx_to_cm(p_dx)

        p_dy = abs(obj.bbox[3] - obj.bbox[1])
        cm_dy = self.convert_pixel_dy_to_cm(p_dy)

        return max(cm_dx, cm_dy)

    def dimensions(self, obj: CXRObject) -> Tuple[float, float]:
        """Returns the dimensions (in cm) of the CXRObject according to major axis x minor axis."""

        p_dx = abs(obj.bbox[2] - obj.bbox[0])
        cm_dx = self.convert_pixel_dx_to_cm(p_dx)

        p_dy = abs(obj.bbox[3] - obj.bbox[1])
        cm_dy = self.convert_pixel_dy_to_cm(p_dy)

        if cm_dx > cm_dy:
            return (cm_dx, cm_dy)
        else:
            return (cm_dy, cm_dx)

    def find(self, object_name: str) -> List[CXRObject]:
        """Find all objects in the image that match the given object_name.

        Parameters
        ----------
        object_name : str
            The name of the object to find.
            supported: ["endotracheal tube", "carina"]

        Returns
        -------
        List[CXRObject]
            A list of CXRObject instances for each object found that matches the object_name.
        """
        object_name = self.normalize_query(object_name)

        found_objects = []

        if self.exists(object_name) is True:
            bboxes = self.cache.get_object(
                self.rid, object_name, self.original_width, self.original_height
            )

            # Add non-empty bboxes to the found objects
            if bboxes is not None:
                for bbox in bboxes:
                    if bbox is not None:
                        found_objects.append(
                            CXRObject(object_name=object_name, bbox=bbox)
                        )

        return found_objects

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.

        Parameters
        ----------
        object_name : str
            The name of the object to find.
            supported: ["endotracheal tube"]
        """
        object_name = self.normalize_query(object_name)
        return self.cache.check_exists(self.rid, object_name)

    def segment(self, region_name: str) -> CXRSegmentation:
        """Returns a segmentation map of the image based on the region specified.

        Parameters
        ----------
        region_name : str
            The name of the region for segmentation.

        Returns
        -------
        CXRSegmentation
            The segmentation map for the specified region.
        """
        region_name = self.normalize_query(region_name)
        segmentation_map = self.cache.get_segmentation(self.rid, region_name)
        cxr_segmentation = CXRSegmentation(
            region_name=region_name, segmentation_map=segmentation_map
        )
        return cxr_segmentation

    def width(self, segmentation: CXRSegmentation) -> float:
        """Returns the width (in cm) of the widest part of the segmentation map."""
        p_dx = segmentation.get_pixel_width()
        cm_dx = self.convert_pixel_dx_to_cm(p_dx)
        return cm_dx

    def height(self, segmentation: CXRSegmentation) -> float:
        """Returns the height (in cm) of the tallest part of the segmentation map."""
        p_dy = segmentation.get_pixel_height()
        cm_dy = self.convert_pixel_dy_to_cm(p_dy)
        return cm_dy

    def within(self, obj: CXRObject, segmentation: CXRSegmentation) -> bool:
        """Check if the object's center is within a region of another map."""
        center = obj.get_center()

        # Scale center coordinates from the original image size to 512x512
        scaled_x = int(center[0] * (512 / self.original_filesize[0]))
        scaled_y = int(center[1] * (512 / self.original_filesize[1]))

        # Ensure coordinates are within bounds
        scaled_x = min(max(scaled_x, 0), 511)
        scaled_y = min(max(scaled_y, 0), 511)

        # Check if the center point falls within a segmented region (non-zero value)
        return None

    def filter(
        self, objects: List[CXRObject], region: CXRSegmentation
    ) -> List[CXRObject]:
        """Returns all objects with their center within the region."""
        return [obj for obj in objects if self.within(obj, region)]
