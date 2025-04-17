# -*- coding: utf-8 -*-
# CNN pipeline: prepare, train, and predict.

import sys
import argparse
import logging

from model.CNNPipeline import Pipeline

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# main
#
# python rf_driver.py options here
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to extract balanced points using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Senegal using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=True,
                        dest='data_csv',
                        help='Path to the data CSV configuration file')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=['preprocess', 'train', 'predict'])

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

    # Initialize pipeline object
    pipeline = Pipeline(args.config_file, args.data_csv)
    logging.info(pipeline.experiment_name)

    # --------------------------------------------------------------------------------
    # prepare step
    # --------------------------------------------------------------------------------
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess()

    # --------------------------------------------------------------------------------
    # train step
    # --------------------------------------------------------------------------------
    if "train" in args.pipeline_step:
        pipeline.train()

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "predict" in args.pipeline_step:
        pipeline.predict()

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
