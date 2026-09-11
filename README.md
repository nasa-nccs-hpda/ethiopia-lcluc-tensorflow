# Ethiopia LCLUC

Land-cover mapping and multi-year compositing from WorldView imagery for Amhara, Ethiopia.

[![DOI](https://zenodo.org/badge/527702332.svg)](https://zenodo.org/badge/latestdoi/527702332)

This repository provides workflows for CNN and GPU random-forest classification, spatial compositing, and reference-point validation. It also includes the **Amhara Land Cover Explorer**, a Google Earth Engine application for comparing land-cover products and inspecting reference data.

## Explore the land-cover products

The [Earth Engine application](app/ethiopia-lcluc.js) displays GSFC 2 m land cover and observation counts for **2009–2016**, **2018–2022**, and **2017–2024**. Comparison layers include Digital Earth Africa Cropland 2019, ESA WorldCover 2020, ESRI Land Cover 2020, GLAD 2020, and Google Dynamic World 2020, alongside canopy height and validation points.

The land-cover products use five classes: **Crop, Tree/Shrub, Grass, Built, and Water**. Layer controls and opacity sliders support comparison; clicking an enabled validation point displays its reference class.

To run the application, open [app/ethiopia-lcluc.js](app/ethiopia-lcluc.js) in the [Earth Engine Code Editor](https://code.earthengine.google.com/) using an account with access to the referenced assets. See the [application guide](docs/APP.md) for asset IDs, class codes, and NoData conventions.

## Installation

Python 3.10 or newer is required. Install the package with its CPU validation dependencies:

```bash
git clone https://github.com/nasa-nccs-hpda/ethiopia-lcluc-tensorflow.git
cd ethiopia-lcluc-tensorflow
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[validation]'
```

This installation supports reference-point validation without a GPU. CNN training and prediction additionally require TensorFlow, `tensorflow_caney`, and `segmentation_models`; compositing requires `vhr_composite` and its geospatial dependencies; GPU random forest requires RAPIDS/cuML. See [workflow setup and examples](docs/USAGE.md).

Imagery, trained model weights, and reference datasets are not bundled with the repository or downloaded by the software. Supply local data paths through configuration files or command-line arguments.

## Validate a land-cover raster

```bash
ethiopia-validate \
  --reference /path/to/reference.gpkg \
  --predictions /path/to/landcover.tif \
  --output-dir output/validation
```

The default reference field is `val_class`, with codes **0 Crop, 1 Tree/Shrub, 2 Grass, 3 Built, 4 Water**. Other schemes can be supplied using `--class-names` and `--label-map` JSON files.

Validation transforms points into the raster's coordinate system and respects its NoData mask. Outputs include sampled points in `points.gpkg` and accuracy, per-class metrics, and a confusion matrix in `metrics.json`. Additional excluded values can be specified with `--ignore-values`. See [validation methods and limitations](docs/USAGE.md#validation-outputs).

## Training and compositing

| Command | Purpose |
| --- | --- |
| `ethiopia-cnn` | Preprocess training data, train a CNN, and predict land cover |
| `ethiopia-rf` | Train and apply a GPU random-forest classifier |
| `ethiopia-composite` | Build scene footprints, extract metadata, and generate composites |
| `ethiopia-validate` | Evaluate categorical rasters against reference points |

Each command provides `--help`. [Configuration examples](examples/) and the [workflow guide](docs/USAGE.md) describe input formats, environment settings, and execution steps. Training configurations may use different class schemes from the five-class application products.

## Repository structure

| Directory | Contents |
| --- | --- |
| [app/](app/) | Earth Engine application |
| [ethiopia_lcluc_tensorflow/](ethiopia_lcluc_tensorflow/) | Processing pipelines and command-line tools |
| [examples/](examples/) | Portable configurations and input-format examples |
| [projects/](projects/) | Experiment configurations, data manifests, and tile lists |
| [notebooks/](notebooks/) | Dataset preparation and exploratory analyses |
| [docs/](docs/) | Application, workflow, and reproducibility documentation |
| [tests/](tests/) | CPU regression tests |
| [legacy/](legacy/) | Earlier implementations retained for reference |

## Development

```bash
python -m pip install -e '.[validation,test]'
python -m pytest -q
ruff check ethiopia_lcluc_tensorflow tests
```

The automated tests cover CPU validation, configuration handling, CLI interfaces, and inference locking. GPU workflows and the Earth Engine application require integration testing in their respective environments.

## Citation

Use the DOI badge above to access the archived software record and its citation information. Guidance on recording data versions, model configurations, and execution environments is provided in the [reproducibility notes](docs/USAGE.md#reproducibility).
