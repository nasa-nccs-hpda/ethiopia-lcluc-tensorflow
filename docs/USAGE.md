# Workflow guide

Install the package as described in the [README](../README.md). Commands below assume the repository root is the working directory. Replace `/path/to/...` placeholders with accessible input and output locations.

## Processing environments

| Workflow | Additional software |
| --- | --- |
| CNN | TensorFlow, `tensorflow_caney`, `segmentation_models`, Rasterio, GeoPandas, Xarray, Rioxarray, scikit-learn |
| Compositing | `vhr_composite`, GDAL, Dask, GeoPandas, NumPy, Pandas, Xarray and supporting geospatial libraries |
| GPU random forest | CuPy, cuDF, cuML, Rasterio, Rioxarray, NumPy, Joblib |

GPU workflows require a compatible NVIDIA/CUDA environment. The `validation` package extra installs CPU validation dependencies; it does not install these external processing stacks. Record the exact environment and upstream software versions for reproducible runs.

A [Dockerfile](../requirements/Dockerfile) is provided for environments based on `nasanccs/vhr-cloudmask`. Supply a compatible, tested base image tag or digest using the `BASE_IMAGE` build argument. Container builds install this repository's source.

## CNN classification

Start with [examples/cnn.yaml](../examples/cnn.yaml):

```bash
export ETHIOPIA_IMAGES=/path/to/imagery
export ETHIOPIA_OUTPUT=/path/to/output
```

Review the bands, normalization, class count, model settings, and GPU devices before running. The example retains the archived v12 six-class experiment settings, selects Blue, Green, Red, and NIR1 from eight input bands, and uses local standardization.

Training manifests contain image paths, label paths, and tile counts:

```csv
data,label,ntiles
/path/to/image.tif,/path/to/label.tif,3000
```

```bash
ethiopia-cnn -c examples/cnn.yaml -d /path/to/training.csv -s preprocess train
ethiopia-cnn -c examples/cnn.yaml -s predict
```

Prediction reads `inference_regex_list` and the configured model-loading settings. Preprocessed tiles are stored beneath `data_dir`, model artifacts beneath `model_dir`, and scene predictions beneath `inference_save_dir`.

Repeated `--set KEY=VALUE` arguments override YAML settings, for example `--set model_dir=/path/to/models --set gpu_devices=0`. The resolved configuration is saved with the model artifacts. Relative paths are interpreted from the working directory.

Output locks prevent concurrent workers from predicting the same scene. Completed rasters are renamed into place after writing. If a process is killed, confirm that its job has stopped before removing a leftover `.lock` file.

### Slurm execution

The [submission script](../slurm/predict.sh) accepts a configuration path and additional CNN CLI arguments:

```bash
export ETHIOPIA_CONTAINER=/path/to/compatible-container.sif
export ETHIOPIA_BINDS=/path/to/data,/path/to/output
sbatch slurm/predict.sh /path/to/experiment.yaml
```

Submit from the repository root, or set `ETHIOPIA_REPO`. Use `ETHIOPIA_PYTHONPATH` for additional source checkouts. Load Singularity if required by the cluster, and select appropriate account, partition, and resource settings with `sbatch` options.

## Reference-point validation

Validate each product separately:

```bash
ethiopia-validate \
  --reference /path/to/reference.gpkg \
  --predictions '/path/to/product/tiles/*.tif' \
  --output-dir output/validation \
  --class-names examples/classes-five.json
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--label-column` | Reference field; defaults to `val_class` |
| `--label-map` | JSON mapping text reference labels to numeric codes |
| `--class-names` | JSON mapping numeric codes to class names |
| `--ignore-values` | Additional prediction values to exclude |
| `--band` | Prediction band; defaults to 1 |
| `--overlap` | Choose `first` or `last` when valid raster predictions overlap; default is an error |
| `--overwrite` | Replace existing validation outputs |

Use `--label-column Land_Use` for the supported text reference labels. For the ESRI comparison asset used in the application, add `--ignore-values 7` to match its display mask.

The CNN CLI also accepts `-s validate` with `-vd`, `--validation-predictions`, and `--validation-output-dir`, or their YAML equivalents `validation_database`, `validation_predictions`, and `validation_output_dir`. Validation-only runs do not load TensorFlow. Use `ethiopia-validate` for custom label and class mappings.

### Validation outputs

`points.gpkg` retains the reference geometries and attributes, with the reference class, sampled prediction, selected raster path, and count of valid overlapping predictions. `metrics.json` records evaluated/excluded sample counts, overall accuracy, balanced accuracy, per-class precision/recall/F1, and a confusion matrix with rows representing reference classes and columns representing predictions. Balanced accuracy is the mean recall over represented reference classes.

Points without a valid prediction are retained in the GeoPackage and excluded from metrics. Overlap selection uses sorted absolute filenames; `first` and `last` do not imply acquisition date or data quality. Validate comparison products separately and use a common sample set when required by the study design. These are unweighted point metrics, not area-adjusted estimates or confidence intervals.

## Compositing

Start with [examples/composite.yaml](../examples/composite.yaml) and configure the data locations:

```bash
export ETHIOPIA_IMAGES=/path/to/imagery
export ETHIOPIA_PREDICTIONS=/path/to/predictions
export ETHIOPIA_CLOUDMASKS=/path/to/cloudmasks
export ETHIOPIA_GRID=/path/to/grid.gpkg
export ETHIOPIA_OUTPUT=/path/to/output
```

Cloud masks are produced externally, for example by `vhr-cloudmask`. Configure the desired period, class definitions, scene filename suffixes, and output products. A tile list contains one grid identifier per line, such as `h00v00`; [example lists](../projects/composite/configs/tile_lists/) are provided.

```bash
ethiopia-composite -c examples/composite.yaml -s build_footprints extract_metadata
ethiopia-composite -c examples/composite.yaml -t /path/to/tiles.txt -s composite
```

Footprints associate imagery with the grid, metadata describes the scenes, and compositing processes the requested tiles. Configurable outputs include mode land cover, observation counts, class-frequency products, and confidence metrics.

The date filter includes January 1 of `start_year` through December 31 of `end_year`. CLI overrides use repeated `--set KEY=VALUE` arguments and are saved in the run configuration. The historical `grid_path` key is accepted as an alias for `grid_filename`.

`post_process_combine` is not implemented and must remain `false`. Historical configurations may require adaptation to the current `vhr_composite` dependency; use the portable template as the starting point.

## GPU random forest

Prepare a CSV containing numeric feature columns followed by the target column `CLASS`, with no missing values. Prediction raster bands must match the training feature order.

```bash
ethiopia-rf --step train --train-csv /path/to/pixel-training.csv \
  --output-model /path/to/model.pkl \
  --train-size 0.80 --n-trees 200 --max-features log2 --seed 42

ethiopia-rf --step predict --output-model /path/to/model.pkl \
  --rasters '/path/to/images/*.tif' \
  --output-dir /path/to/predictions --window-size 5120
```

`--seed` controls shuffling, splitting, and the RF estimator. The CLI also provides `vis` for inspecting a saved model. Dataset preparation examples are available in the [notebooks](../notebooks/); the RF CLI does not implement preprocessing.

## Reproducibility

Record the software commit, resolved configuration, data versions, class mappings, model checkpoints, normalization statistics, and training/validation split for each analysis. Preserve the Python dependency versions, upstream processing-library commits, and GPU/CUDA or container version. Evaluation should account for spatial and temporal sampling design; the five-class application scheme may differ from historical training configurations.
