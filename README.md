
# 🩻 FactCheXcker

[[CVPR Paper (coming soon!)]]()
[[Arxiv Paper]](https://arxiv.org/abs/2411.18672)
[[Notebook example]](https://github.com/rajpurkarlab/FactCheXcker/blob/main/factchexcker_examples.ipynb)

This repository contains code to run the FactCheXcker pipeline on model-generated chest X-ray reports that contain quantifiable metrics, such as endotracheal tube placements.

## ⚙️ Pipeline

<div style="display: flex;">
	<img src="https://github.com/rajpurkarlab/FactCheXcker/blob/main/assets/pipeline-square.png" height="500" />
	<img src="https://github.com/rajpurkarlab/FactCheXcker/blob/main/assets/pipeline-example.png" height="500"/>
</div>


FactCheXcker has three main components: Query Generator, Code Generator, and Report Updater. When provided with a medical image and its corresponding model-generated report that may contain hallucinated measurements, the Query Generator identifies potential measurement discrepancies in the report, the Code Generator creates and executes specialized code to obtain measurements from the image, and the Report Updater integrates the new measurements into the report.

## 🏗️ Setup

**Step 1**: First, use the command below to make a copy of the repository.

```bash
git clone git+https://github.com/rajpurkarlab/FactCheXcker
```

**Step 2:** Create and install the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

**Step 3:** Update `llm.py` to use the LLM of your choosing. In the paper, we use  the Azure OpenAI wrapper and thus must update the endpoint constants in `contants.py` and set the API key on the command line using `export AZURE_OPENAI_API_KEY=<your api key here>`.

## 🗂️ Pipeline Overview

FactCheXcker employs plug-and-play tool modules to update quantifiable measurements in a medical report. These tools are then dynamically called by the three steps of the pipeline, namely
- `Query Generator`: Based on a report, identifies measurable findings and generates natural language queries to perform re-measurements.
- `Code Generator`: Based on a query, uses the provided API – implemented in `api.py` – to generate code that solves the task at hand. The API relies on a few core functions that through executable code can be combined to achieve more complex tasks. These are the following:
	-  `exists(object_name: str)`: Returns `True` if the object can be found in the image, `False` otherwise.
	- `find(object_name: str)`: Returns a list of CXRObjects centered around any objects found in the image matching the object name.
	- `segment(region_name: str)`: Returns a segmentation map (CXRSegmentation) of the image based on the region specified.
- `Report Updater`: Updates the report using the new measurements. 

The FactCheXcker is built to allow adding and editing which model backbones (e.g. ResNets, custom fine-tuned models, rule-based approach) are called when the pipeline calls, for example, the `exists` and `find` functions on specific objects.

## 💻 Python usage

### Example 1: Using the FactCheXcker Pipeline
The following Python code first creates a `CXRImage` instance and then runs the full pipeline on it using FactCheXcker.

```python
from FactCheXcker import FactCheXcker
from api import *

# Config file specifies which tools to use for what commands
config_path = "configs/config.json"
module_registry = CXRModuleRegistry(config_path)

cxr_image = CXRImage(
    rid="image00001",
    image_path="data/image00001.jpg",
    report="The ETT is positioned 3.2 cm above the carina",
    original_size=[1200, 1200],
    pixel_spacing=(0.139, 0.139),
    module_registry=module_registry,
)

# Use FactCheXcker to automatically update the report
pipeline = FactCheXcker(config_path)
updated_report = pipeline.run_pipeline(cxr_image)
```

### Example 2: Using the API Directly
The following Python code combines `find` and `exist` calls to find the distance between the endotracheal tube and carina in a chest X-ray image.

```python
from api import *

# Config file specifies which tools to use for what commands 
module_registry = CXRModuleRegistry("configs/config.json")

# Load an image and report
cxr_image = CXRImage(
    rid="image00001",
    image_path="data/image00001.jpg",
    report="The ETT is positioned 3.2 cm above the carina",
    original_size=[1200, 1200],
    pixel_spacing=(0.139, 0.139),
    module_registry=module_registry,
)

# Check if ETT present, then get distance between ETT and carina
ett_present = cxr_image.exists("ett")
if (ett_present):
	print("[*] ETT identified.")
	ett_objects = cxr_image.find("ett")
	carina_objects = cxr_image.find("carina")
	distance_cm = cxr_image.distance(ett_objects[0], carina_objects[0])
	print(f"[*] The ETT is {distance_cm:.2f} cm from the carina.")
	
```
## 📒 Notebook example

See the notebook `factchexcker_examples.ipynb` for code samples on a few real chest X-ray images.

## 🛠️ Adding New Tools

Please follow these steps to add, edit, and delete tools to be used in the FactCheXcker pipeline.

**Step 1: Add/Modify a Config File**
 The `configs` folder contains configurations specifying which tools/models to use for specific functions in the API. The configuration below, for instance, specifies that for the **command** `find` on the **object** string `ett`, call the `find_ett_carinanet`function. If an object call is not present, the pipeline will return an error that it cannot handle that task.
 
```json
// configs/config.json
{
	"find": {
		"ett": "tools.find.find_ett_carinanet",
		"carina": "tools.find.find_carina_carinanet"
	},
	"exists": {
		"ett": "tools.exists.exists_ett_resnet"
	}
}
```

The files `tools/find.py` and `tools/exists.py`, for instance, contain implementations of the `find` and `exists` commands presented in the paper using a fine-tuned `CarinaNet` and `ResNet50+`, respectively.

Let's say you want to use a module to implement the `find('lesion')` command. Begin by adding the following in your `config.json` file:
```json
// configs/config.json
{
	"find": {
		...
		"lesion": "tools.find.find_lesion_modelname"
	}
}
```

**Step 2: Import the model in the corresponding `tools/<command>.py` file**

Following our example above, in the `tools/find.py` file you would first (if necessary) place your model in the `models/` directory and load in your model at the top of the file. If using a plug-and-play module from, for example, HuggingFace, you would only have to import and load it.

```python
# tools/find.py

# Model from models/ directory
from models.LesionModel import LesionModel
model_lesion = LesionModel()

# E.g. Model from HuggingFace
import transformers
model_lesion = transformers.from_pretrained("model")
```

**Step 3: Implement the tool under the tools/<command\>.py files**

In the same `tools/find.py` file, create a new function that takes in a `CXRImage` object with metadata about the Chest X-ray image/report and the queries `object_name` as input, and outputs a list of `CXRObject`'s. **Make sure to use the provided `rescale` functions to rescale the model outputs to the original image dimensions**. Please see the `api.py` file for documentation of the properties and methods of `CXRImage` and `CXRObject`.

```python
def find_lesion_modelname(cxr_image: CXRImage, object_name: str):
    """Find Lesion locations in an image."""
    resized_dim = 224
    preds = model_lesion(cxr_image.image_path, resized_dim)
    point_prediction = preds["ett"]
    rescaled = cxr_image.rescale_point(
        point_prediction,
        (resized_dim, resized_dim),
    )
    return [cxr_image.point_to_bbox(rescaled)]
```

**Step 4: Use your tool**!

Now your new tool is ready to use! The API prompt given to FactCheXcker will automatically update with the available options based on the configuration you use. The following code snippet would call the newly created tool in a python script:

```python
from api import *

# Specify which tools to use for what commands 
module_registry = CXRModuleRegistry("configs/config.json")

# Load an image and report
cxr_image = CXRImage(
    rid="image00001",
    image_path="data/image00001.jpg",
    report="The lesion measures 3.1 x 2.3 cm",
    original_size=[1200, 1200],
    pixel_spacing=(0.139, 0.139),
    module_registry=module_registry,
)

# Use our new tool!
lesion_objects = cxr_image.find("lesion")
```

## 🔖 License

FactCheXcker's pipeline code is released under the MIT License. See [LICENSE]() for further details.
