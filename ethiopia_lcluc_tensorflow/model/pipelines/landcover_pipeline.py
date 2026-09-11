import os
import re
import sys
import time
import logging
from contextlib import ExitStack
from ethiopia_lcluc_tensorflow.utils.locking import output_lock
import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import tensorflow as tf
import tensorflow_caney as tfc
import segmentation_models as sm
from rioxarray.merge import merge_arrays

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from multiprocessing import Pool, cpu_count

from tensorflow_caney.model.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.utils.data import gen_random_tiles, \
    get_dataset_filenames, get_mean_std_dataset

from tensorflow_caney.utils.data import read_dataset_csv, \
    gen_random_tiles, modify_bands, normalize_image, rescale_image, \
    modify_label_classes, get_dataset_filenames, get_mean_std_dataset, \
    get_mean_std_metadata, read_metadata
# from vhr_cnn_chm.model.atl08 import ATL08
# from tensorflow_caney.utils.vector.extract import \
#    convert_coords_to_pixel_location, extract_centered_window
# from tensorflow_caney.utils.data import modify_bands, \
#    get_dataset_filenames, get_mean_std_dataset, get_mean_std_metadata
# from tensorflow_caney.utils.system import seed_everything
# from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
# from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
# from tensorflow_caney.utils import indices
# from tensorflow_caney.utils.model import load_model
# from tensorflow_caney.inference import regression_inference
# from pygeotools.lib import iolib, warplib

from tensorflow_caney.utils.model import load_model, get_model
from tensorflow_caney.utils import indices

# osgeo.gdal.UseExceptions()

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow_caney.model.pipelines.cnn_segmentation import CNNSegmentation
from tensorflow_caney.inference import inference

class LandCoverPipeline(CNNSegmentation):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename, data_csv=None, logger=None):

        # Configuration file intialization
        self.conf = self._read_config(config_filename, Config)

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Set Data CSV
        self.data_csv = data_csv

        # Set experiment name
        try:
            self.experiment_name = self.conf.experiment_name.name
        except AttributeError:
            self.experiment_name = self.conf.experiment_name

        # output directory to store metadata and artifacts
        # self.metadata_dir = os.path.join(self.conf.data_dir, 'metadata')
        # self.logger.info(f'Metadata dir: {self.metadata_dir}')

        # Set output directories and locations
        # self.intermediate_dir = os.path.join(
        #    self.conf.data_dir, 'intermediate')
        # self.logger.info(f'Intermediate dir: {self.intermediate_dir}')

        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        logging.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        logging.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        logging.info(f'Model dir: {self.labels_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # save configuration into the model directory
        OmegaConf.save(self.conf, os.path.join(self.model_dir, 'config.yaml'))

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self) -> None:

        logging.info('Starting prediction stage')

        first_strategy = tf.distribute.MirroredStrategy()
        with first_strategy.scope():
            # Load model for inference
            model = load_model(
                model_filename=self.conf.model_filename,
                model_dir=self.model_dir,
                conf=self.conf,
                custom_objects={
                    '_iou': sm.metrics.iou_score,
                    'iou_score': sm.metrics.iou_score,
                    'focal_tversky_loss': tfc.utils.losses.focal_tversky_loss,
                    'binary_tversky_loss': tfc.utils.losses.binary_tversky_loss,
                    'focal_loss_plus_dice_loss': sm.losses.categorical_focal_dice_loss
                }
            )

        # Retrieve mean and std, there should be a more ideal place
        if self.conf.standardization in ["global", "mixed"]:
            mean, std = get_mean_std_metadata(
                os.path.join(
                    self.model_dir,
                    f'mean-std-{self.conf.experiment_name}.csv'
                )
            )
            logging.info(f'Mean: {mean}, Std: {std}')
        else:
            mean = None
            std = None

        # gather metadata
        if self.conf.metadata_regex is not None:
            metadata = read_metadata(
                self.conf.metadata_regex,
                self.conf.input_bands,
                self.conf.output_bands
            )

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            # set output directory
            basename = os.path.basename(os.path.dirname(filename))
            if basename == 'M1BS' or basename == 'P1BS':
                basename = os.path.basename(
                    os.path.dirname(os.path.dirname(filename)))

            output_directory = os.path.join(
                self.conf.inference_save_dir, basename)
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # Acquire atomically; release on failed reads, skips, and exceptions.
            with output_lock(output_filename) as acquired, ExitStack() as sources:
                if not acquired:
                    logging.info('Skipping existing or locked output %s', output_filename)
                    continue

                try:

                    logging.info(f'Starting to predict {filename}')

                    # if metadata is available
                    if self.conf.metadata_regex is not None:

                        # get timestamp from filename
                        year_match = re.search(
                            r'(\d{4})(\d{2})(\d{2})', filename)
                        timestamp = str(int(year_match.group(2)))

                        # get monthly values
                        mean = metadata[timestamp]['median'].to_numpy()
                        std = metadata[timestamp]['std'].to_numpy()
                        self.conf.standardization = 'global'

                    # open filename
                    image = sources.enter_context(rxr.open_rasterio(filename))
                    logging.info(f'Prediction shape: {image.shape}')

                    # check bands in imagery, do not proceed if one band
                    if image.shape[0] == 1:
                        logging.info(
                            'Skipping file because of non sufficient bands')
                        continue

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                # Calculate indices and append to the original raster
                image = indices.add_indices(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)

                # Modify the bands to match inference details
                image = modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                logging.info(
                    f'Prediction min={image.min().values}, ' +
                    f'max={image.max().values}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                temporary_tif = xr.where(image > -100, image, 600)

                print("ENTERING PREDICTION")

                # Sliding window prediction
                prediction = \
                    inference.sliding_window_tiler_multiclass(
                        xraster=temporary_tif,
                        model=model,
                        n_classes=self.conf.n_classes,
                        overlap=self.conf.inference_overlap,
                        batch_size=self.conf.pred_batch_size,
                        threshold=self.conf.inference_treshold,
                        standardization=self.conf.standardization,
                        mean=mean,
                        std=std,
                        normalize=self.conf.normalize,
                        rescale=self.conf.rescale,
                        window=self.conf.window_algorithm,
                        probability_map=self.conf.probability_map
                    )

                if isinstance(prediction, tuple):
                    prediction, probability = prediction

                print('Prediction output: ', np.unique(prediction))

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                )

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction.attrs['model_name'] = (self.conf.model_filename)
                prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(
                    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                partial_filename = output_filename + '.partial.tif'
                try:
                    prediction.rio.to_raster(
                        partial_filename,
                        BIGTIFF="IF_SAFER",
                        compress=self.conf.prediction_compress,
                        driver=self.conf.prediction_driver,
                        dtype=self.conf.prediction_dtype
                    )
                    os.replace(partial_filename, output_filename)
                finally:
                    Path(partial_filename).unlink(missing_ok=True)
                del prediction

                print(Path(output_filename).with_suffix('.tif'))

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

        return

    def validate(self, validation_database=None, **options):
        """Validate configured predictions with the standalone CPU validator."""
        from ethiopia_lcluc_tensorflow.model.validation import validate_points
        reference = validation_database or self.conf.get("validation_database")
        if not reference:
            raise ValueError("Specify a validation database.")
        predictions = options.pop("predictions", None) or self.conf.get("validation_predictions")
        if not predictions:
            raise ValueError("Specify validation_predictions; prediction selection must be explicit.")
        output_dir = options.pop("output_dir", None) or self.conf.get("validation_output_dir")
        if not output_dir:
            raise ValueError("Specify validation_output_dir.")
        return validate_points(reference, predictions, output_dir, **options)
