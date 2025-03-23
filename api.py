from typing import List, Tuple, Optional, Callable, Dict
import h5py
import torch
import math
import re
import importlib
import json


class CXRModuleRegistry:
    """Registry for dynamically loading and storing user-defined find and exists functions."""

    def __init__(self, config_path: str):
        self.commands: Dict[str, Dict[str, Callable]] = {
            "find": {},
            "exists": {},
            "segment": {},
        }
        self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load user-defined function mappings from a JSON config."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load command functions
        for command in self.commands.keys():
            for object_name, func_path in config.get(command, {}).items():
                if func_path:
                    self.commands[command][object_name] = self._load_function(func_path)

    def _load_function(self, full_function_path: str) -> Callable:
        """Dynamically load a function from a module path."""
        module_name, func_name = full_function_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def get_command_function(self, command: str, object_name: str) -> Callable:
        """Retrieve a command function for the given object."""
        command_functions = self.commands.get(command, None)
        if not command_functions:
            return None
        command_func = command_functions.get(object_name, None)
        return command_func


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
        Id of CXR Image object
    report: str
        Generated model report for the image.
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
        report: str,
        original_size: List[int],
        pixel_spacing: Tuple[float, float],
        module_registry: CXRModuleRegistry,
        # cache: ModuleCache,
    ):
        self.rid = rid
        self.image_path = image_path
        self.report = report
        self.original_width, self.original_height = original_size
        self.pixel_spacing = pixel_spacing
        self.module_registry = module_registry
        # self.cache = cache

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

    def rescale_point(
        self,
        point: Tuple[float, float],
        curr_dim: Tuple[float, float],
    ) -> float:
        """Rescale predicted image value"""
        rescaled = (
            self.rescale_x(point[0], self.original_width, curr_dim[0]),
            self.rescale_y(point[1], self.original_height[1], curr_dim[1]),
        )
        return rescaled

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
        find_func = self.module_registry.get_command_function("find", object_name)
        if find_func is None:
            print(f"[Warning] No 'find' function registered for '{object_name}'")
            return []
        found_bboxes = find_func(self)  # Function should return a list of bbox tuples
        return [CXRObject(object_name=object_name, bbox=bbox) for bbox in found_bboxes]

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.

        Parameters
        ----------
        object_name : str
            The name of the object to find.
            supported: ["endotracheal tube"]
        """
        object_name = self.normalize_query(object_name)
        exists_func = self.module_registry.get_command_function("exists", object_name)
        if exists_func is None:
            print(f"[Warning] No 'exists' function registered for '{object_name}'")
            return False
        return exists_func(self)

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
