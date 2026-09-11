# Ethiopia LCLUC

WorldView land-cover mapping and multi-year compositing for Amhara, Ethiopia, with a Google Earth Engine (GEE) app for exploring the resulting products and comparison datasets.

[![DOI](https://zenodo.org/badge/527702332.svg)](https://zenodo.org/badge/latestdoi/527702332)

The repository contains TensorFlow CNN preprocessing, training, and prediction workflows; a GPU random-forest workflow; spatial compositing; and the Amhara Land Cover Explorer. Training and compositing depend on external geospatial software and project data on NASA NCCS Explore/ADAPT. The GEE app runs independently against uploaded Earth Engine assets.

## Repository guide

| Location | Contents |
| --- | --- |
| [app/ethiopia-lcluc.js](app/ethiopia-lcluc.js) | GEE explorer, product controls, legends, and validation-point inspection |
| [ethiopia_lcluc_tensorflow/view/](ethiopia_lcluc_tensorflow/view/) | CNN, random-forest, and compositing command-line entry points |
| [ethiopia_lcluc_tensorflow/model/pipelines/](ethiopia_lcluc_tensorflow/model/pipelines/) | CNN prediction/validation and compositing implementations |
| [projects/landcover/config/experiments/](projects/landcover/config/experiments/) | Dated experiment YAML files and training-data CSV manifests |
| [projects/composite/configs/](projects/composite/configs/) | Development/production configuration examples and tile lists |
| [notebooks/](notebooks/) | Dataset preparation, exploratory analysis, and single-/multi-GPU random-forest experiments |
| [requirements/Dockerfile](requirements/Dockerfile) | Container definition based on `nasanccs/vhr-cloudmask` |
| [slurm/predict.sh](slurm/predict.sh) | Site-specific GPU inference submission example |
| [docs/](docs/) | Historical dataset, random-forest, and project notes |
| [legacy/](legacy/) | Earlier CNN implementations, scripts, and configurations |
| `data/` | Local rasters and validation vectors; ignored by Git and not supplied by a clone |
| [tests/](tests/) | CPU validation, configuration, CLI, and inference-locking regression tests |
| [examples/](examples/) | Portable CNN/composite YAML templates and class/manifest examples |

## Amhara Land Cover Explorer

### Run and update the app

1. Open the [Earth Engine Code Editor](https://code.earthengine.google.com/) with an account and project that can read the assets below.
2. Copy [app/ethiopia-lcluc.js](app/ethiopia-lcluc.js) into a script, save it, and run it.
3. Check the layers and validation-point popup in the Code Editor.
4. To update a hosted app, publish the saved script through Earth Engine's Apps interface. Editing this repository does not update a deployed app automatically. Ensure that the app itself can read the required assets; see the [Earth Engine Apps documentation](https://developers.google.com/earth-engine/guides/apps).

The app uses Earth Engine's `ee`, `ui`, and `Map` globals. It is not a standalone browser or Node.js application.

### Explore the products

The study boundary and **GSFC LCLU 2017–2024** are enabled initially. Other products are available through checkboxes:

- GSFC 2 m land cover for **2009–2016**, **2018–2022**, and **2017–2024**.
- Observation counts for the same three periods.
- Aligned/reclassified Digital Earth Africa Cropland 2019, ESA WorldCover 2020, ESRI Land Cover 2020, GLAD 2020, and Google Dynamic World 2020.
- Meta Canopy Height 1 m and the 2026 validation/reference points.

Comparison rasters sit beneath the GSFC land-cover layers. Enable one comparison at a time and reduce the **LCLU 2017–2024** opacity or turn off the overlying land-cover layers to see it. The shared five-class legend explicitly covers GSFC, ESA, ESRI, GLAD, and Dynamic World. Cropland extent, observation count, canopy height, and reference points have separate legends.

Enable **Validation / Reference Points 2026**, then click a point to open a bottom-left popup with its class name, numeric `val_class`, and original `Land_Use` label. The nearest point within an eight-pixel click tolerance is highlighted in yellow. Close dismisses the popup; disabling the points through the control panel also clears the selection.

### Classes and NoData

These codes describe the final five-class app products. Historical training datasets and six-class model configurations can use different schemes; check the specific experiment before reusing labels.

| Code | Class | Display color |
| --- | --- | --- |
| 0 | Crop | `#ffaa00` |
| 1 | Tree / Shrub | `#267300` |
| 2 | Grass | `#ffffbe` |
| 3 | Built | `#730000` |
| 4 | Water | `#0070ff` |

Local raster inspection found the following values. Validation-point sampling supports the interpretation of codes 0–4, but the GeoTIFFs do not contain embedded class names.

| Comparison product | Displayed values | Masked values in the app |
| --- | --- | --- |
| Digital Earth Africa Cropland 2019 | 0 = crop | 255 (non-crop/declared NoData) |
| ESA WorldCover 2020 | 0–4 | −128 |
| ESRI Land Cover 2020 | 0–4 | 7 and 15 |
| GLAD 2020 | 0–4; this reclassified file is not binary | 15 |
| Google Dynamic World 2020 | 0–4 | 15 |

ESRI code 7 is treated as NoData for display, in addition to the file's declared NoData value of 15. These masks do not modify the source GeoTIFFs. Crop code 0 remains visible.

### Earth Engine assets

The app reads the following assets under `projects/gsfc-dsg/assets/`:

| Product | Asset name |
| --- | --- |
| LCLU 2009–2016 | `Amhara_LCLU_5class_2009_2016_2m_native_cog_clean_cog` |
| LCLU 2018–2022 | `Amhara_LCLU_5class_2018_2022_2m_native_cog_clean_cog` |
| LCLU 2017–2024 | `Amhara_LCLU_5class_2017_2024_2m_native_cog_clean_cog` |
| Observations 2009–2016 | `Amhara_nobservations_2009_2016_2m_native_cog_clean_cog` |
| Observations 2018–2022 | `Amhara_nobservations_2018_2022_2m_native_cog_clean_cog` |
| Observations 2017–2024 | `Amhara_nobservations_2017_2024_2m_native_cog_clean_cog` |
| Digital Earth Africa | `DigitalEarthAfrica_crop_mask_2019_Amhara_LCLUcrop0_nonCrop255` |
| ESA WorldCover | `ESA_WorldCover_10m_2020_v100_Amhara_reclass` |
| ESRI | `ESRI_LULC_36P37P_2020_Amhara_reclass` |
| GLAD | `GLAD2020_Amhara_reclass` |
| Dynamic World | `Google_DynamicWorld_LULC_2020_mode_2_reclass` |
| Validation points | `Amhara_validation_points_2026` |

Additional dependencies are the boundary asset `projects/ee-jacaraba-ethiopia/assets/boundaries/Amhara_Study_Area_Boundary_4buf10km_EPSG_GEE`, canopy-height collection `projects/sat-io/open-datasets/facebook/meta-canopy-height`, and palette module `users/gena/packages:palettes`.

The comparison images use their first band. Validation points are loaded as a FeatureCollection and filtered to the study boundary. The original ESA WorldCover catalog layer and GLAD Cropland 2019 layer have been replaced by the project assets above.

## Python environment and data

Clone the repository and run Python commands from its root:

```bash
git clone https://github.com/nasa-nccs-hpda/ethiopia-lcluc-tensorflow.git
cd ethiopia-lcluc-tensorflow
```

Use a project-compatible Linux geospatial environment, with NVIDIA/CUDA support for the GPU workflows. The source imports these main dependencies:

| Workflow | Dependencies |
| --- | --- |
| CNN | `tensorflow`, `tensorflow_caney`, `segmentation_models`, OmegaConf, Rasterio, GeoPandas, NumPy, Xarray, Rioxarray, scikit-learn |
| Compositing | `vhr_composite`, GDAL, Dask, OmegaConf, GeoPandas, NumPy, Pandas, Xarray and the supporting geospatial stack |
| GPU random forest | CuPy, cuDF, cuML, GDAL, Rasterio, Xarray, Pandas, NumPy, Joblib |

Install the package and CPU validation dependencies with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[validation,test]'
ethiopia-validate --help
```

The installed commands are `ethiopia-cnn`, `ethiopia-composite`, `ethiopia-rf`, and `ethiopia-validate`. Help works without GPU libraries. CNN, RF, and compositing execution still require the compatible external stacks listed above; they are deliberately not installed by the lightweight validation extra. Record the tested upstream commits and GPU environment for a publication run. See [publication and reproducibility notes](docs/PUBLICATION.md).

The previous project setup documented this Explore container path:

```text
/explore/nobackup/projects/ilab/containers/ethiopia-lcluc-tensorflow.2025.04
```

Confirm its availability and dependencies on the cluster before use. For a configured container, a GPU invocation can use this pattern (replace the paths):

```bash
ethiopia_repo="$PWD"
ethiopia_container=/path/to/compatible-container.sif
singularity exec --nv \
  --bind "$ethiopia_repo:/workspace",/explore/nobackup/projects,/lscratch \
  --pwd /workspace --env PYTHONPATH=/workspace \
  "$ethiopia_container" \
  python -m ethiopia_lcluc_tensorflow.view.landcover_cnn_pipeline_cli --help
```

Bind any additional locations referenced by your configuration. If `tensorflow_caney` or `vhr_composite` are external source checkouts rather than installed packages, bind those directories and include their container paths in `PYTHONPATH`.

Source data, model checkpoints, and generated products are not downloaded automatically. Existing configurations reference locations such as:

| Data | Example project location |
| --- | --- |
| WorldView TOA scenes | `/explore/nobackup/projects/hls/EVHR/Amhara-MS/` |
| Land-cover predictions | `/explore/nobackup/projects/3sl/development/cnn_landcover/` |
| Cloud masks | `/explore/nobackup/projects/3sl/products/cloudmask/v2/` |
| Auxiliary grids | `/explore/nobackup/projects/3sl/auxiliary/Shapefiles/` |
| Evaluation rasters and validation database | `/panfs/ccds02/nobackup/projects/3sl/auxiliary/Ethiopia/` |

These are site-specific locations, not public download links. Update configuration paths and output directories for your environment.

## CNN land-cover workflow

The [CNN CLI](ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py) exposes `preprocess`, `train`, `predict`, and `validate`. Preprocessing and training are inherited from `tensorflow_caney`; the local pipeline implements prediction and validation.

Start with [examples/cnn.yaml](examples/cnn.yaml), derived from [the archived v12 configuration](projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v12.yaml). Set `ETHIOPIA_OUTPUT` and `ETHIOPIA_IMAGES` to your output and imagery directories. Review `data_dir`, `model_dir`, input/output bands, `n_classes`, normalization/standardization, model settings, GPU devices, and inference paths. This example has six output classes and selects Blue, Green, Red, and NIR1 from eight input bands. Despite its filename, its `standardization` setting is `local`; use YAML contents as the source of truth.

Training manifests use the header `data,label,ntiles`, for example:

```csv
data,label,ntiles
/path/to/image.tif,/path/to/label.tif,3000
```

After preparing a configuration and manifest:

```bash
python -m ethiopia_lcluc_tensorflow.view.landcover_cnn_pipeline_cli \
  -c /path/to/experiment.yaml \
  -d /path/to/training.csv \
  -s preprocess train

python -m ethiopia_lcluc_tensorflow.view.landcover_cnn_pipeline_cli \
  -c /path/to/experiment.yaml \
  -s predict
```

Prediction uses `inference_regex_list` and model-loading settings from the configuration. It writes scene predictions beneath `inference_save_dir`; preprocessing artifacts go beneath `data_dir`, and model artifacts go beneath `model_dir`.

Override any YAML setting with repeated `--set KEY=VALUE` arguments, for example `--set model_dir=/path/to/models --set gpu_devices=0`. Resolved settings are saved with the model artifacts. Prediction uses exclusive output locks, closes input rasters, and writes to a temporary raster before renaming a completed output. A killed job can leave a lock; confirm the job has stopped before removing it.

### Validate categorical products on a CPU

Run each product separately with explicit raster paths or quoted globs:

```bash
ethiopia-validate \
  --reference data/Amhara_validation_database__jun2026_tilecorrected_tilealigned.gpkg \
  --predictions data/GLAD2020_Amhara_reclass.tif \
  --output-dir output/validation/glad2020
```

For ESRI, add `--ignore-values 7` to match the app's additional NoData mask. The default reference field is `val_class`; use `--label-column Land_Use` for text labels. `--class-names examples/classes-five.json` explicitly selects the five-class scheme; supply another JSON mapping for other schemes. `--label-map` accepts a JSON mapping from text labels to class codes.

The validator writes sampled `points.gpkg` and `metrics.json`, including evaluated/excluded counts, confusion matrix, accuracy, and per-class metrics. It respects each raster's CRS and masks, and rejects overlapping predictions unless you choose `--overlap first` or `--overlap last`. Existing results require `--overwrite`. See [metric definitions and limitations](docs/PUBLICATION.md#validation-outputs).

The CNN CLI also supports `-s validate` with `-vd`, `--validation-predictions`, and `--validation-output-dir` (or the corresponding `validation_database`, `validation_predictions`, and `validation_output_dir` YAML settings), without loading TensorFlow. Use the standalone validator for custom label/class mappings. The GEE point popup displays reference attributes; it does not run accuracy assessment.

[slurm/predict.sh](slurm/predict.sh) accepts a configuration path and uses `ETHIOPIA_CONTAINER`, optional `ETHIOPIA_REPO`, `ETHIOPIA_BINDS`, and `ETHIOPIA_PYTHONPATH` environment variables. Submit from the repository root; set partition/account and resource requests for your cluster. Load Singularity before submission if your site requires a module.

```bash
export ETHIOPIA_CONTAINER=/path/to/compatible-container.sif
export ETHIOPIA_BINDS=/path/to/data,/path/to/output
sbatch slurm/predict.sh /path/to/experiment.yaml
```

## Cloud masks and compositing

Cloud-mask generation is handled by the external `vhr-cloudmask` software. This repository consumes those masks alongside land-cover predictions; it does not provide a cloud-mask generation CLI.

Use [examples/composite.yaml](examples/composite.yaml) as a starting point. Set `ETHIOPIA_IMAGES`, `ETHIOPIA_PREDICTIONS`, `ETHIOPIA_CLOUDMASKS`, `ETHIOPIA_GRID`, and `ETHIOPIA_OUTPUT` for your environment. Set the imagery, grid, land-cover, cloud-mask, metadata, and output locations, plus the desired years and class definitions. A tile-list file contains one grid tile ID per line, such as `h00v00`; examples are in [tile_lists/](projects/composite/configs/tile_lists/).

Run the stages in order:

```bash
python -m ethiopia_lcluc_tensorflow.view.landcover_composite_pipeline_cli \
  -c /path/to/composite.yaml \
  -s build_footprints extract_metadata

python -m ethiopia_lcluc_tensorflow.view.landcover_composite_pipeline_cli \
  -c /path/to/composite.yaml \
  -t projects/composite/configs/tile_lists/amhara_tiles_0.txt \
  -s composite
```

`build_footprints` associates imagery with the grid; `extract_metadata` prepares scene metadata; `composite` processes the requested tiles. Depending on configuration, outputs include mode land-cover composites, observation counts, class-frequency products, and confidence metrics, with GeoPackage intermediates and logs.

The current local compositing date filter includes January 1 of `start_year` through December 31 of `end_year`. Some older production YAML comments describe an exclusive upper bound; those comments do not match this filter. The loader accepts the historical `grid_path` alias for `grid_filename` and supplies defaults for optional filtering/output flags. Repeated `--set KEY=VALUE` arguments override YAML values and are saved in the run snapshot. Unimplemented `post_process_combine=true` now raises an error instead of silently doing nothing; the portable example disables it. Review upstream `vhr_composite` compatibility before running older configurations.

## GPU random forest

The [RF CLI](ethiopia_lcluc_tensorflow/view/landcover_rf_pipeline_cli.py) supports `train`, `predict`, and `vis` using RAPIDS/cuML. Its training CSV must contain numeric feature columns followed by the target column named `CLASS`, with no missing values. Raster bands must match the training feature order.

```bash
python -m ethiopia_lcluc_tensorflow.view.landcover_rf_pipeline_cli \
  --step train --train-csv /path/to/pixel-training.csv \
  --output-model /path/to/model.pkl \
  --train-size 0.80 --n-trees 200 --max-features log2

python -m ethiopia_lcluc_tensorflow.view.landcover_rf_pipeline_cli \
  --step predict --output-model /path/to/model.pkl \
  --rasters '/path/to/images/*.tif' \
  --output-dir /path/to/predictions --window-size 5120
```

The `--seed` option controls shuffling, splitting, and the RF estimator. Prediction reads rasters through Rioxarray; training requires a compatible RAPIDS environment. Dataset preparation and related experiments are in [EthiopiaDatasetGen.ipynb](notebooks/EthiopiaDatasetGen.ipynb), [EthiopiaRandomForest.ipynb](notebooks/EthiopiaRandomForest.ipynb), and [EthiopiaRandomForestMultiGPU.ipynb](notebooks/EthiopiaRandomForestMultiGPU.ipynb). The RF CLI does not expose a preprocessing step.

## Checks and historical documentation

For a local syntax/whitespace check of app edits:

```bash
node --check app/ethiopia-lcluc.js
git diff --check
```

These checks do not validate Earth Engine UI properties, asset access, or server-side queries. Run the app in Earth Engine and check product toggles, NoData transparency, legends, and validation clicks before publishing. Python workflows require their external dependencies and project data; this README's examples have been checked against the source interfaces, not executed as end-to-end HPC runs. Run the CPU regression suite and package checks with:

```bash
python -m pytest -q
ruff check ethiopia_lcluc_tensorflow tests
python -m build
python -m twine check dist/*
```

GitHub/GitLab CI runs CPU tests and package builds. GPU training and compositing still need integration checks in the project environment. The Dockerfile installs this checkout, and publishing workflows require manual dispatch. Before release, complete the [publication metadata and environment record](docs/PUBLICATION.md#release-information-still-required).

Historical references remain available in [dataset notes](docs/DATASET.md), [random-forest notes](docs/README-RandomForest.md), [project notes](docs/README.md), and [the model survey](docs/ModelSurvey.md). Their paths, scripts, and class schemes may predate the current workflows. Use the DOI badge above to locate the repository's archived release record.
