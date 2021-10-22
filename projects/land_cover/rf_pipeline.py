# -*- coding: utf-8 -*-
"""
CPU and GPU Random Forest Pipeline - preprocess, train, predict.
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
import rasterio as rio
import rasterio.features as riofeat

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as sklRFC

from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

try:
    import cupy as cp
    import cudf as cf
    from cuml.ensemble import RandomForestClassifier as cumlRFC
    from cuml.dask.ensemble import RandomForestClassifier as cumlRFC_mg
    from cupyx.scipy.ndimage import median_filter
    cp.random.seed(seed=None)
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

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


def toraster(
        rast: str, prediction: np.array, nodataval=[-9999],
        dtype: str = 'int16', output: str = 'rfmask.tif'):
    """
    Save tif file from numpy to disk.
    :param rast: raster name to get metadata from
    :param prediction: numpy array with prediction output
    :param dtype type to store mask on
    :param output: raster name to save on
    :return: None, tif file saved to disk
    ----------
    Example
        raster.toraster(filename, raster_obj.prediction, outname)
    ----------
    """
    # get meta features from raster
    with rio.open(rast) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    # logging.info(meta)

    prediction = prediction.astype('int16')

    nodatavals[nodatavals == 0] = -9999
    prediction[nodatavals == -9999] = \
        nodatavals[nodatavals == -9999]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = dtype  # data type modification

    # write to a raster
    with rio.open(output, 'w', **out_meta) as dst:
        dst.write(prediction, 1)
    logging.info(f'Prediction saved at {output}')


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
        '--gpu', dest='has_gpu', action='store_true', default=False)

    parser.add_argument(
        '--output-model', type=str, required=False,
        dest='output_pkl', help='Path to the output PKL file (.pkl)')

    parser.add_argument(
        '--data-csv', type=str, required=False,
        dest='data_csv', help='Path to the data CSV configuration file')

    parser.add_argument(
        '--train-csv', type=str, required=False,
        dest='train_csv', help='Path to the output CSV file')

    parser.add_argument(
        '--bands', type=str, nargs='*', required=False,
        dest='bands', help='Bands to store in CSV file',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2'])

    parser.add_argument(
        '--step', type=str, nargs='*', required=True,
        dest='pipeline_step', help='Pipeline step to perform',
        default=['preprocess', 'train', 'predict', 'vis'],
        choices=['preprocess', 'train', 'predict', 'vis'])

    parser.add_argument(
        '--seed', type=int, required=False, dest='seed',
        default=42, help='Random SEED value')

    parser.add_argument(
        '--test-size', type=float, required=False,
        dest='test_size', default=0.20, help='Test size rate (e.g .30)')

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
    # preprocess step
    # --------------------------------------------------------------------------------
    if "preprocess" in args.pipeline_step:

        # ----------------------------------------------------------------------------
        # 1. Read data csv file
        # ----------------------------------------------------------------------------
        assert os.path.exists(args.data_csv), f'{args.data_csv} not found.'
        data_df = pd.read_csv(args.data_csv)  # initialize data dataframe
        assert not data_df.isnull().values.any(), f'NaN found: {args.data_csv}'
        logging.info(f'Open {args.data_csv} for preprocessing.')

        # ----------------------------------------------------------------------------
        # 2. Extract points out of spatial imagery - rasters
        # ----------------------------------------------------------------------------
        # set empty dataframe to store point values
        df_points = pd.DataFrame(columns=args.bands + ['CLASS'])
        list_points = []
        logging.info(f"Generating {data_df['ntiles'].sum()} points dataset.")

        # start iterating over each file
        for di in data_df.index:

            # get filename for output purposes, in the future, add to column
            filename = data_df['data'][di].split('/')[-1]
            logging.info(f'Processing {filename}')

            # read imagery from disk and process both image and mask
            img = xr.open_rasterio(data_df['data'][di]).values
            mask = xr.open_rasterio(data_df['label'][di]).values

            # ------------------------------------------------------------------------
            # Unique processing of this project - Start
            # ------------------------------------------------------------------------
            # squeeze mask if needed, start classes fro 0-n_classes
            mask = np.squeeze(mask) if len(mask.shape) != 2 else mask
            mask = mask - 1 if np.min(mask) == 1 else mask
            # ------------------------------------------------------------------------
            # Unique processing of this project - Done
            # ------------------------------------------------------------------------

            # crop ROI, from outside to inside based on pixel address
            ymin, ymax = data_df['ymin'][di], data_df['ymax'][di]
            xmin, xmax = data_df['xmin'][di], data_df['xmax'][di]
            img, mask = \
                img[:, ymin:ymax, xmin:xmax], mask[ymin:ymax, xmin:xmax]

            # crop ROI, from outside to inside based on pixel value
            # img = np.clip(img, 0, 10000)

            # get N points from imagery
            points_per_class = data_df['ntiles'][di] // args.n_classes
            logging.info(f'Generating {points_per_class} points per class.')

            # extract values from imagery, two classes
            for cv in range(args.n_classes):

                logging.info(f'Starting with class: {cv}')
                bbox = img.shape  # size of the image
                counter = 0  # counter for class balancing

                while counter < points_per_class:

                    # get indices and extract spectral and class value
                    y, x = random.randrange(bbox[1]), random.randrange(bbox[2])
                    sv, lv = img[:, y, x], int(mask[y, x])

                    if lv == cv:
                        # trying speed up here - looks like from list is faster
                        list_points.append(
                            pd.DataFrame(
                                [np.append(sv, [lv])],
                                columns=list(df_points.columns))
                        )
                        counter += 1

        df_points = pd.concat(list_points)

        # ----------------------------------------------------------------------------
        # 3. Save file to disk
        # ----------------------------------------------------------------------------
        df_points.to_csv(args.train_csv, index=False)
        logging.info(f'Saved dataset file {args.train_csv}')

    # --------------------------------------------------------------------------------
    # train step
    # --------------------------------------------------------------------------------
    if "train" in args.pipeline_step:

        # ----------------------------------------------------------------------------
        # 1. Read data csv file
        # ----------------------------------------------------------------------------
        assert os.path.exists(args.train_csv), f'{args.train_csv} not found.'
        data_df = pd.read_csv(args.train_csv, sep=',')
        assert not data_df.isnull().values.any(), f'Na found: {args.train_csv}'
        logging.info(f'Open {args.train_csv} dataset for training.')

        # ----------------------------------------------------------------------------
        # 2. Shuffle and Split Dataset
        # ----------------------------------------------------------------------------
        data_df = data_df.sample(frac=1).reset_index(drop=True)  # shuffle data

        # split dataset, fix type
        x = data_df.iloc[:, :-1].astype(np.float32)
        y = data_df.iloc[:, -1].astype(np.int8)

        # split data into training and test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.seed)
        del data_df, x, y

        # logging some of the model information
        logging.info(f'X size: {x_train.shape}, Y size:  {y_train.shape}')
        logging.info(f'ntrees={str(args.n_trees)}, maxfeat={args.max_feat}')

        # ------------------------------------------------------------------
        # 3. Instantiate RandomForest object - FIX this area
        # ------------------------------------------------------------------
        if args.has_gpu:  # run using RAPIDS library

            # initialize cudf data and log into GPU memory
            logging.info('Training model via RAPIDS.')

            # single gpu setup
            x_train = cf.DataFrame.from_pandas(x_train)
            x_test = cf.DataFrame.from_pandas(x_test)
            y_train = cf.Series(y_train.values)
            rf_funct = cumlRFC  # RF Classifier

            # TODO: multi gpu setup
            # https://github.com/rapidsai/cuml/blob/branch-21.12/notebooks/random_forest_mnmg_demo.ipynb
            # cluster = LocalCUDACluster(
            # threads_per_worker=1, n_workers=n_workers)
            # c = Client(cluster)
            # workers = c.has_what().keys()
            # rf_funct = cumlRFC_mg

            # Shard the data across all workers
            # X_train_df, y_train_df = dask_utils.persist_across_workers(
            # c,[X_train_df,y_train_df],workers=workers)

            # Build and train the model
            # cu_rf_mg = cuRFC_mg(**cu_rf_params)
            # cu_rf_mg.fit(X_train_df, y_train_df)

            # Check the accuracy on a test set
            # cu_rf_mg_predict = cu_rf_mg.predict(X_test)
            # acc_score = accuracy_score(
            # cu_rf_mg_predict, y_test, normalize=True)
            # c.close()
            # cluster.close()

        else:
            logging.info('Training model via SKLearn.')
            rf_funct = sklRFC

        # Initialize model
        rf_model = rf_funct(
            n_estimators=args.n_trees, max_features=args.max_feat)

        # ------------------------------------------------------------------
        # 4. Fit Model
        # ------------------------------------------------------------------
        # fit model to training data and predict for accuracy score
        rf_model.fit(x_train, y_train)

        if args.has_gpu:
            acc_score = accuracy_score(
                y_test, rf_model.predict(x_test).to_array())
            p_score = precision_score(
                y_test, rf_model.predict(x_test).to_array(), average='macro')
            r_score = recall_score(
                y_test, rf_model.predict(x_test).to_array(), average='macro')
            f_score = f1_score(
                y_test, rf_model.predict(x_test).to_array(), average='macro')
        else:
            acc_score = accuracy_score(y_test, rf_model.predict(x_test))
            p_score = precision_score(y_test, rf_model.predict(x_test), average='macro')
            r_score = recall_score(y_test, rf_model.predict(x_test), average='macro')
            f_score = f1_score(y_test, rf_model.predict(x_test), average='macro')

        logging.info(f'Test Accuracy:  {acc_score}')
        logging.info(f'Test Precision: {p_score}')
        logging.info(f'Test Recall:    {r_score}')
        logging.info(f'Test F-Score:   {f_score}')

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
                riofeat.sieve(prediction, 800, prediction, None, 8)

                # median
                prediction = median_filter(cp.asarray(prediction), size=20)
                prediction = cp.asnumpy(prediction)

                toraster(
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

        logging.info(dir(model))   # .estimators_[0])
        #    export_text(
        #        model.estimators_[0], spacing=3, decimals=3,
        #        feature_names=args.bands)
        #    )

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
