# -*- coding: utf-8 -*-
"""
GPU Random Forest Pipeline - preprocess, train, predict.
Author: Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
"""
import os
import gc
import sys
import glob
import random
import logging
import argparse
from tqdm import tqdm

import joblib
import numpy as np
import xarray as xr
import pandas as pd
from osgeo import gdal, osr
import rasterio as rio
import rasterio.features as riofeat

import cupy
import cudf
import cuml

from cuml.ensemble import RandomForestClassifier as cumlRFC
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
from cupyx.scipy.ndimage import median_filter

cupy.random.seed(seed=24)

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# rf_driver.py methods
# -----------------------------------------------------------------------------
def predict(data, model, ws=[5120, 5120]):
    """
    Predict from model.
    :param data: raster xarray object
    :param model: loaded model object
    :param ws: window size to predict on
    :return: prediction output in numpy format
    ----------
    Example
        raster.toraster(filename, raster_obj.prediction, outname)
    ----------
    """
    # open rasters and get both data and coordinates
    rast_shape = data[0, :, :].shape  # shape of the wider scene
    wsx, wsy = ws[0], ws[1]  # in memory sliding window predictions

    # if the window size is bigger than the image, predict full image
    if wsx > rast_shape[0]:
        wsx = rast_shape[0]
    if wsy > rast_shape[1]:
        wsy = rast_shape[1]

    prediction = np.zeros(rast_shape)  # crop out the window
    logging.info(f'wsize: {wsx}x{wsy}. Prediction shape: {prediction.shape}')

    for sx in tqdm(range(0, rast_shape[0], wsx)):  # iterate over x-axis
        for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
            x0, x1, y0, y1 = sx, sx + wsx, sy, sy + wsy  # assign window
            if x1 > rast_shape[0]:  # if selected x exceeds boundary
                x1 = rast_shape[0]  # assign boundary to x-window
            if y1 > rast_shape[1]:  # if selected y exceeds boundary
                y1 = rast_shape[1]  # assign boundary to y-window

            window = data[:, x0:x1, y0:y1]  # get window
            window = window.stack(z=('y', 'x'))  # stack y and x axis
            window = window.transpose("z", "band").values  # reshape

            # perform sliding window prediction
            prediction[x0:x1, y0:y1] = \
                model.predict(window).reshape((x1 - x0, y1 - y0))
    # save raster
    return prediction.astype('int16')  # type to int16


def to_cog(
        input_array, output_filename, original_filename,
        transform, epsg=32628, ndval=255, ovr=[2, 4, 8, 16, 32, 64]):

    # get geospatial profile, will apply for output file
    with rio.open(original_filename) as src:
        nodatavals = src.read_masks(1).astype('int16')
    input_array[nodatavals != ndval] = ndval

    pixel_width, offset, x_origin, offset, pixel_height, y_origin = transform
    x_size, y_size = input_array.shape

    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', y_size, x_size, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(
        (x_origin, pixel_width, offset, y_origin, offset, pixel_height))

    dataset.GetRasterBand(1).WriteArray(input_array)
    dataset.GetRasterBand(1).SetNoDataValue(ndval)
    dataset.BuildOverviews("NEAREST", ovr)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    dataset.SetProjection(outRasterSRS.ExportToWkt())

    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(
        output_filename, dataset,
        options=["COPY_SRC_OVERVIEWS=YES", "TILED=YES", "COMPRESS=LZW"])
    del input_array, dataset
    return


# -----------------------------------------------------------------------------
# main
#
# python rf_driver.py options here
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Random Forest Segmentation pipeline for tabular and spatial data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--output-model', type=str, required=False,
        dest='output_pkl', help='Path to the output PKL file (.pkl)')

    parser.add_argument(
        '--train-csv', type=str, required=False,
        dest='train_csv', help='Path to the output CSV file')

    parser.add_argument(
        '--step', type=str, nargs='*', required=True,
        dest='pipeline_step', help='Pipeline step to perform',
        default=['train', 'predict', 'vis'],
        choices=['train', 'predict', 'vis'])

    parser.add_argument(
        '--seed', type=int, required=False, dest='seed',
        default=42, help='Random SEED value')

    parser.add_argument(
        '--train-size', type=float, required=False,
        dest='train_size', default=0.80, help='Train size rate (e.g .30)')

    parser.add_argument(
        '--n-trees', type=int, required=False,
        dest='n_trees', default=20, help='Number of trees (e.g 20)')

    parser.add_argument(
        '--max-features', type=str, required=False,
        dest='max_feat', default='log2', help='Max features (e.g log2)')

    parser.add_argument(
        '--n-classes', type=int, required=False,
        dest='n_classes', default=2, help='Number of classes to predict')

    parser.add_argument(
        '--rasters', type=str, required=False, dest='rasters',
        default='*.tif', help='rasters to search for')

    parser.add_argument(
        '--window-size', type=int, required=False,
        dest='ws', default=5120, help='Prediction window size (e.g 5120)')

    parser.add_argument(
        '--output-dir', type=str, required=False,
        dest='output_dir', default='', help='output directory')

    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # set logging
    # --------------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    ch.setLevel(logging.INFO)

    # set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------------------------------------------------------------------------------
    # train step
    # --------------------------------------------------------------------------------
    if "train" in args.pipeline_step:

        # ----------------------------------------------------------------------------
        # 1. Read data csv file
        # ----------------------------------------------------------------------------
        assert os.path.exists(args.train_csv), f'{args.train_csv} not found.'
        data_df = cudf.read_csv(args.train_csv, sep=',')
        assert not data_df.isnull().values.any(), f'Na found: {args.train_csv}'
        logging.info(f'Open {args.train_csv} dataset for training.')

        # ----------------------------------------------------------------------------
        # 2. Shuffle and Split Dataset
        # ----------------------------------------------------------------------------
        data_df = data_df.sample(frac=1).reset_index(drop=True)  # shuffle data
        logging.info(data_df['CLASS'].value_counts())

        # split dataset, fix type
        x = data_df.iloc[:, :-1].astype(np.float32)
        y = data_df.iloc[:, -1].astype(np.int8)

        # split data into training and test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=args.train_size)
        del data_df, x, y

        # logging some of the model information
        logging.info(f'X size: {x_train.shape}, Y size:  {y_train.shape}')
        logging.info(f'ntrees={str(args.n_trees)}, maxfeat={args.max_feat}')

        # ------------------------------------------------------------------
        # 3. Instantiate RandomForest object
        # ------------------------------------------------------------------
        rf_model = cumlRFC(
            n_estimators=args.n_trees, max_features=args.max_feat)

        # fit model to training data and predict for accuracy score
        rf_model.fit(x_train, y_train)

        # ------------------------------------------------------------------
        # 4. Predict test set for accuracy metrics
        # ------------------------------------------------------------------
        acc_score = accuracy_score(y_test, rf_model.predict(x_test).to_array())
        logging.info(f'Test Accuracy:  {acc_score}')

        # make output directory
        os.makedirs(
            os.path.dirname(os.path.realpath(args.output_pkl)), exist_ok=True)

        # export model to file
        try:
            joblib.dump(rf_model, args.output_pkl)
            logging.info(f'Model has been saved as {args.output_pkl}')
        except Exception as e:
            logging.error(f'ERROR: {e}')

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "predict" in args.pipeline_step:

        assert os.path.exists(args.output_pkl), f'{args.output_pkl} not found.'
        model = joblib.load(args.output_pkl)  # loading pkl in parallel
        logging.info(f'Loaded model {args.output_pkl}.')

        # create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # 3b3. apply model and get predictions
        rasters = glob.glob(args.rasters)
        assert len(rasters) > 0, "No raster found"
        logging.info(f'Predicting {len(rasters)} files.')

        for rast in rasters:  # iterate over each raster

            filename = rast.split('/')[-1]
            output_filename = os.path.join(args.output_dir, filename)

            if not os.path.isfile(output_filename):

                gc.collect()  # clean garbage
                logging.info(f"Starting new prediction...{rast}")
                img = xr.open_rasterio(rast)
                logging.info(f'Modified image: {img.shape}')

                # crop ROI, from outside to inside based on pixel value
                img = np.clip(img, 0, 10000)
                prediction = predict(img, model, ws=[args.ws, args.ws])

                # sieve
                #riofeat.sieve(prediction, 800, prediction, None, 8)

                # median
                #prediction = median_filter(cp.asarray(prediction), size=20)
                #prediction = cp.asnumpy(prediction)

                to_cog(
                    rast=rast, prediction=prediction, output=output_filename)
                prediction = None  # unload between each iteration

            else:
                logging.info(f'{output_filename} already predicted.')

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "vis" in args.pipeline_step:

        assert os.path.exists(args.output_pkl), f'{args.output_pkl} not found.'
        model = joblib.load(args.output_pkl)  # loading pkl in parallel
        logging.info(f'Loaded model {args.output_pkl}.')
        logging.info(dir(model))

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
