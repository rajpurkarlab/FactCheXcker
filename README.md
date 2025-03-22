# FactCheXcker

[[Paper]](https://arxiv.org/abs/2411.18672)
[[Notebook example]]()

This repository contains code to run the FactCheXcker pipeline on model-generated chest X-ray reports that contain quantifiable metrics, such as endotracheal tube placements.

## Pipeline

![Pipeline]()

FactCheXcker has three main components: Query Generator, Code Generator, and Report Updater. When provided with a medical image and its corresponding model-generated report that may contain hallucinated measurements, the Query Generator identifies potential measurement discrepancies in the report, the Code Generator creates and executes specialized code to obtain measurements from the image, and the Report Updater integrates the new measurements into the report.

## Setup

The following command will pull and install the latest commit from this repository, along with its Python dependencies:

```bash
git clone git+https://github.com/rajpurkarlab/FactCheXcker
```

## Files

Below is a description of the key files in the repository:

## Command-line usage

## Python usage

## Notebook examples

## License

FactCheXcker's pipeline code is released under the MIT License. See [LICENSE]() for further details.
