import os
import gc
import sys
import time
import tqdm
import shutil
import logging
import argparse
import omegaconf
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from osgeo import gdal
from pathlib import Path
from dask.diagnostics import ProgressBar

from vhr_composite.model.composite import Composite
from vhr_composite.model.footprints import Footprints
from vhr_composite.model.metadata import Metadata
from vhr_composite.model.utils import TqdmLoggingHandler
from vhr_composite.model import post_process
from vhr_composite.model.metrics import (
    calculate_trend_confidence_numba,
    calculate_distribution_confidence_numba
)

class LandCoverCompositePipeline(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename: str):

        # Configuration file intialization
        self.conf = omegaconf.OmegaConf.load(config_filename)

        # select output directory based on filter name
        if self.conf.filter_months:
            self.conf.output_tif_dir = os.path.join(
                self.conf.output_dir,
                f'{self.conf.start_year}.{self.conf.end_year}.' +
                self.conf.filter_name
            )
        else:
            self.conf.output_tif_dir = os.path.join(
                self.conf.output_dir,
                f'{self.conf.start_year}.{self.conf.end_year}'
            )

        # create output directory
        os.makedirs(self.conf.output_tif_dir, exist_ok=True)

        # set logging
        self._set_logging(
            f'{self.conf.start_year}.{self.conf.end_year}')

        logging.info(f'Created output dir: {self.conf.output_tif_dir}')

        # copy config file to output directory
        shutil.copy(config_filename, self.conf.output_tif_dir)
        logging.info(f'Saved config file to: {self.conf.output_tif_dir}')

    # -------------------------------------------------------------------------
    # _set_logging
    # -------------------------------------------------------------------------
    def _set_logging(self, description: str = ''):
        """
        Set logging configuration.
        """
        # setup logging configuration
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        fh_name = os.path.join(
            self.conf.output_tif_dir,
            f'{description}.log')

        # set file handler logging configuration
        ch = logging.FileHandler(fh_name)
        sh = TqdmLoggingHandler()
        ch.setLevel(logging.INFO)
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(sh)

        logging.info(f'Output logs sent to: {fh_name}')
        return logger

    # -------------------------------------------------------------------------
    # _run_accelerated_function
    # -------------------------------------------------------------------------
    def _run_accelerated_function(
                self,
                data_array,
                function,
                input_core_dims: str = 'time'
            ):

        # Apply over (time) dimension for each (y, x)
        result = xr.apply_ufunc(
            function,
            data_array,
            input_core_dims=[[input_core_dims]],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[np.float32]
        )

        # Result is a 2D DataArray with shape (y, x)
        # Trigger computation and monitor progress
        with ProgressBar():
            result.compute()

        # do it in parallel for speed for more files
        return result.expand_dims(dim='band', axis=0)

    # -------------------------------------------------------------------------
    # build_footprints
    # -------------------------------------------------------------------------
    def build_footprints(self) -> None:

        # set logging
        self._set_logging('build_footprints')

        # create object for building footprint
        footprints = Footprints(
            self.conf,
            input_data=self.conf.input_data_regex,
            output_filename=self.conf.footprints_filename,
            grid_filename=self.conf.grid_filename
        )

        # start the generation of footprints
        footprints.build_footprints()

        return

    # -------------------------------------------------------------------------
    # extract_metadata
    # -------------------------------------------------------------------------
    def extract_metadata(self) -> None:

        # set logging
        self._set_logging('extract_metadata')

        # create object for building footprint
        metadata = Metadata(
            self.conf,
            input_filename=self.conf.footprints_filename,
            output_filename=self.conf.metadata_filename
        )

        # start the generation of metadata
        metadata.extract_metadata()

        return

    # -------------------------------------------------------------------------
    # composite
    # -------------------------------------------------------------------------
    def composite(self, tiles_filename: str) -> None:

        # verify of tiles_filename is not None, else exit
        if tiles_filename is None:
            sys.exit('Specify tiles_filename option as an argument.')

        # set logging
        self._set_logging(
            f'composite-{Path(tiles_filename).stem}')

        # make sure tiles_filename exists
        assert os.path.exists(tiles_filename), \
            f'{tiles_filename} does not exist.'

        # Get gdf with strips of interest
        assert os.path.exists(self.conf.metadata_filename), \
            f'{self.conf.metadata_filename} does not exist. ' + \
            f'You need to run the metadata step before composite.'
        metadata_gdf = gpd.read_file(self.conf.metadata_filename)
        logging.info(f'Reading in metadata {self.conf.metadata_filename}')

        # Some soil moisture values are NaN's, we
        # do not question why, we just fix. This is the way.
        if 'soilM_median' in metadata_gdf.columns:
            soil_m_median = metadata_gdf['soilM_median'].values
            soil_m_median = np.nan_to_num(soil_m_median, nan=9999.0)
            metadata_gdf['soilM_median'] = soil_m_median
            logging.info('Fixing soil moisture data for NaNs.')
        else:
            # adding all scenes as good since we do not have soil moisture data
            metadata_gdf['soilM_median'] = 2799

        # Set landcover outputs as columns in geodataframe
        logging.info('Adding landcover column to metadata.')
        if 'landcover' in self.conf.composite_fields:
            metadata_gdf['landcover'] = list(
                map(lambda f: os.path.join(
                    self.conf.landcover_dir,
                    f'{f}{self.conf.landcover_suffix}.tif'
                ), metadata_gdf['strip_id'])
            )
        else:
            metadata_gdf['landcover'] = np.nan

        # Set cloudmask outputs as columns in geodataframe
        logging.info('Adding cloudmask column to metadata.')
        if 'cloudmask' in self.conf.composite_fields:
            metadata_gdf['cloudmask'] = list(
                map(lambda f: os.path.join(
                    self.conf.cloudmask_dir,
                    f'{f}{self.conf.cloudmask_suffix}.tif'
                ), metadata_gdf['strip_id'])
            )
        else:
            metadata_gdf['cloudmask'] = np.nan

        # We're using acquisition time as our unique identifier,
        # I know, I know, what happens when two sats capture at the
        # same exact time down to the nanosecond? That is between this
        # code and God. (It'll fail)
        metadata_gdf['datetime'] = \
            pd.to_datetime(metadata_gdf['acq_time'])
        unique_gpk_id = Path(tiles_filename).stem
        logging.info(f'Unique gpkg id: {unique_gpk_id}')

        # Store model output gpkg into disk
        model_output_gdf_name = os.path.join(
            self.conf.output_dir,
            f'{self.conf.region}-{self.conf.model_name}-' +
            f'{self.conf.test_name}-{self.conf.grid_cell_name_pre_str}-' +
            f'{unique_gpk_id}.gpkg'
        )
        metadata_gdf.to_file(model_output_gdf_name, driver="GPKG")
        logging.info(f'Saved updated metadata file to {model_output_gdf_name}')

        # Initialize composite class
        composite = Composite(
            name=self.conf.test_name,
            grid_geopackage_path=self.conf.grid_filename,
            model_output_geopackage_path=model_output_gdf_name,
            output_dir=self.conf.output_dir
        )

        # read clean metadata file
        if not os.path.exists(model_output_gdf_name):
            sys.exit(
                f'Metadata filename {model_output_gdf_name} ' +
                'does not exist. Perhaps you need to run the ' +
                'build_footprints step?'
            )
        metadata_gdf = gpd.read_file(model_output_gdf_name)
        logging.info(f'Metadata includes {metadata_gdf.shape[0]} strips.')
        logging.info(f'Years in metadata: {metadata_gdf.year.unique()}')

        # Filter the strips to use in mode and bad data filling to be within
        # the desired "epoch" or year range of the acquisition time
        start_date = pd.to_datetime(
            f"{self.conf.start_year}-01-01").tz_localize('UTC')
        end_date = pd.to_datetime(
            f"{self.conf.end_year}-12-31 23:59:59").tz_localize('UTC')

        # set initial datetime mask
        datetime_mask = \
            (metadata_gdf[self.conf.datetime_column] >= start_date) & \
            (metadata_gdf[self.conf.datetime_column] <= end_date)

        # other filters - currently just months
        if self.conf.filter_months:

            # Extract the month from the datetime column
            metadata_gdf['month'] = metadata_gdf[
                self.conf.datetime_column].dt.month

            # Define the valid months (November to May)
            remove_months = self.conf.remove_months
            logging.info(f'Removing months: {remove_months}')

            # Add the month condition to the datetime_mask
            datetime_mask &= ~metadata_gdf['month'].isin(
                remove_months)

        # set the new datemask if filters were applied
        metadata_gdf_filtered = metadata_gdf[datetime_mask]

        logging.info(
            'Strips remaining after filters: ' +
            f'{metadata_gdf_filtered.shape[0]}.')
        logging.info(
            f'Years remaining: {metadata_gdf_filtered.year.unique()}')

        # Read in batched tile list (100 tiles per file)
        # most regions usually occupy just 10 or so of these
        # 100-tile text file lists.
        logging.info(f'Reading tiles from: {tiles_filename}')
        with open(tiles_filename, 'r') as fh:
            tiles = fh.readlines()
            tiles = [tile.strip() for tile in tiles]
        assert len(tiles) > 0, f'No tiles found in {tiles_filename}'
        logging.info(
            f'Tiles provided in {Path(tiles_filename).stem}: {len(tiles)}')

        # Making this multi-threaded, will it mess with the Numba
        # parallel mode calculations???
        for tile in tqdm.tqdm(tiles):

            # get the number of strips per tile
            len_strips = len(
                metadata_gdf[metadata_gdf['tile'] == tile])
            logging.info(f'Number of strips in {tile}: {len_strips}')

            if len_strips < 1:
                continue

            # name of the output file
            output_name = f'{self.conf.grid_cell_name_pre_str}.{tile}'

            # set the output filename
            output_tile_filename = os.path.join(
                self.conf.output_dir, f'{output_name}.zarr')
            logging.info(f'Processing {output_tile_filename}')

            # output of mode, dont need second region name
            # add var to more easily replace for other files
            if self.conf.filter_months:
                mode_name = f'{self.conf.filter_name}.mode'
            else:
                mode_name = 'mode'

            # set mode output basename
            mode_bname = \
                f'{self.conf.start_year}.{self.conf.end_year}' + \
                f'.{mode_name}'

            # set output filename for mode
            output_mode_filename = os.path.join(
                self.conf.output_tif_dir,
                f'{output_name}.{mode_bname}.tif')
            msg = \
                'Storing output .tif like: ' + \
                f"{output_mode_filename.replace('mode', '*')}"
            logging.info(msg)

            # generate single grid information
            tile_grid_dataset = composite.generate_single_grid(
                tile,
                write_out=True,
                overwrite=self.conf.overwrite_zarrs,
                fill_value=self.conf.fill_value,
                burn_area_value=self.conf.burn_area_value,
                grid_cell_name_pre_str=self.conf.grid_cell_name_pre_str
            )

            if not tile_grid_dataset:
                continue

            # MW changes for when zarr is read from disk
            # (not sure why this is suddenly a problem)
            # fillna and compute does nothing for grid
            # generated above but fixes issue with zarr from read
            tile_grid_data_array = tile_grid_dataset[output_name].fillna(
                  tile_grid_dataset._FillValue).compute().astype(np.uint32)

            # maybe not necessary but if read from zarr, attrs not in dataarray
            tile_grid_data_array.attrs.update(tile_grid_dataset.attrs)

            del tile_grid_dataset

            # print(tile_grid_data_array)
            # *MW this takes the filtered tile gdf
            # (eg currently the epoch gdf) and
            # gets only the single tile gdf -
            # would we filter for other attributes here?
            metadata_per_tile_fltrd = \
                metadata_gdf_filtered[metadata_gdf_filtered['tile'] == tile]

            len_filtered_strips = len(metadata_per_tile_fltrd)
            logging.info(
                f'Number of filtered strips in {tile}: {len_filtered_strips}')

            # if no strips are left, continue
            if len_filtered_strips < 1:
                continue

            # *MW do we need this anymore, or should we just filter based on X
            # attributes eg soil moisture, date, whatever, and accept the
            # possibility of holes (that we can fill later
            # with post-processing)? Use the updated GDF to further filter by
            # soil moisture QA
            # TODO: enter value threshold
            if self.conf.soil_moisture_qa:
                logging.info('QAin with soil moisture > ')
                good, bad = self.soilMoistureQA(metadata_per_tile_fltrd)
            else:
                logging.info('Not QAing with soil moisture')
                good = metadata_per_tile_fltrd.copy()
                bad = pd.DataFrame(columns=metadata_per_tile_fltrd.columns)

            logging.info(
                f'Paths for filtered strips:\n {good["toa_path"].values}')

            # When filling holes we want to start with the "best" of the bad
            # i.e. the lowest soil moisture first
            bad = bad.sort_values(by='soilM_median')

            # if adding other filters besides years, they just need to be in
            # the "good" dataframe, which they will be if they are in
            # metadata_per_tile_fltrd & do not get removed by soil moisture QA
            passed_qa_datetimes = list(good.datetime.values)
            not_passed_qa_datetimes = list(bad.datetime.values)

            # what is the point of this? added try to avoid issue
            # with missing outputs
            try:
                tile_grid_data_array.sel(time=passed_qa_datetimes)
                tile_grid_data_array.sel(time=not_passed_qa_datetimes)
            except KeyError as ke:  # this happens when strip is missing
                logging.info(f'Problem with slicing array with time: {ke}')

                # find missing datetimes
                tile_grid_times = \
                    pd.to_datetime(tile_grid_data_array.time.values)
                remove_dts = [
                    dt for dt in passed_qa_datetimes
                    if dt not in tile_grid_times]

                # Decide whether to process tiles without missing/corrupt
                # inputs and add to list, or skip tile entirely
                # TODO only do this duinrg debug? otherwise, skip tile and print
                if True: # if debug On, remove and print removed datetimes. Save tile to list to reprocess

                    # include only passed_qa_datetimes that are present in tile_grid_data_array.time
                    valid_dts = [dt for dt in passed_qa_datetimes \
                                                  if dt in tile_grid_times]
                    passed_qa_datetimes = valid_dts
                    reprocess_list = os.path.join(self.conf.output_dir, 
                                                  f'_reprocess-tiles.txt') 
                    msg = 'Removed datetimes from list and added tile to'+\
                                f' list: {remove_dts} ==> {reprocess_list}'
                    logging.info(msg)
                    with open(reprocess_list, 'a') as of: 
                        of.write(f'{tile},{remove_dts}\n')

                    # Check new length of passed datetimes *MW
                    if len(passed_qa_datetimes) < 1:
                        logging.info(f'0 passed datetimes after removal. Skipping {tile}')
                        continue

                else: # do not debug
                    logging.info(
                        f'Tile {tile} not processed due to missing landcover output(s): {remove_dts}')
                    continue

            logging.info(
                f'Continuing with {len(passed_qa_datetimes)} datetimes ' +
                f'for {tile} (of {len(passed_qa_datetimes)+len(not_passed_qa_datetimes)})'
            )

            # this should no longer fail due to if/else above
            tile_grid_ds_good = tile_grid_data_array.sel(
                time=passed_qa_datetimes)
            tile_grid_ds_bad = tile_grid_data_array.sel(
                time=not_passed_qa_datetimes)

            nodata_value = np.uint8(self.conf.fill_value)

            output_mode_filename = Path(output_mode_filename)

            # TODO: currently
            # set = tile_grid_data_array to not exclude "bad qa" datetimes in calculations (not nobs which always uses both as of now)
            tile_grid_use = tile_grid_ds_good
            del tile_grid_data_array

            overwrite = self.conf.overwrite_tifs 

            # write stack from grid to temp .tif for debugging 
            # TODO - add debug param in config?
            # write stack to disk
            if False:
                # import pdb;pdb.set_trace()
                landcover_stack = tile_grid_use.rio.write_nodata(nodata_value)

                lc_stack_filename = str(
                    output_mode_filename).replace('.mode.tif', '.landcover-temp.tif')
                landcover_stack.squeeze().rio.to_raster(lc_stack_filename,
                                                            dtype = np.uint8,
                                                            compress='lzw')
                del landcover_stack

            # below calls now use the boolean calculate parameters from .yaml 
            # to determine which outputs should be created by calling the various functions
            if self.conf.calculate_mode_composite: 
                logging.info('Reducing with multi-mode')
                #*MW why is this called as function instead of method on composite obj?
                #*MW call should be more or less the same, but with diff variabls - NOTE gpu=False (seg fault issues?)
                reduced_stack = Composite.reduce_stack('multi_mode',
                                                    tile_grid_use,
                                                    output_mode_filename,
                                                    overwrite=overwrite,
                                                    nodata=nodata_value,
                                                    gpu=True,
                                                    # gpu=False,  #*MW
                                                    )

                if reduced_stack is not None: #* this will return None if final output already made

                    logging.info('Filling holes with best of the "bad" tiles')
                    #*MW what is this doing? seems to be doing the QA (eg filling 
                    # holes with 'bad' soilmoisture in order from best to worst of 
                    # the bad - aka it's doing nothing right now but keep anyways) - update log for more info
                    # not doing anything eg  np.all(reduced_stack_hole_filled == reduced_stack).values ==> array(True) [but this could also be due to no holes in the first place]
                    reduced_stack_hole_filled = post_process.fill_holes(
                                                        tile_grid_ds_bad,
                                                        reduced_stack,
                                                        not_passed_qa_datetimes,
                                                        nodata_value,
                                                        logging)
                    #*MW temp - should be no 0s
                    if 0 in reduced_stack_hole_filled.data:
                        import pdb; pdb.set_trace()
                    
                    #*MW - edits to fix warp vs nonwarp issues (changing filename; writing nodata; passing compress correctly)
                    # actually set nodata value before writing to tif - TODO - make nodata values consistent
                    reduced_stack_hole_filled.rio.write_nodata(nodata_value,
                                                                   inplace=True)
                    unwarped_tile = str(output_mode_filename).replace('.tif', 
                                                                    '-temp.tif')
                    logging.info(f'Writing unwarped to .tif: {str(unwarped_tile)}') 
                    reduced_stack_hole_filled.rio.to_raster(str(unwarped_tile),
                                                            # dtype=np.uint32,
                                                            dtype = np.uint8,
                                                            compress='lzw')
                    
                    creationOptions = ['COMPRESS=LZW']
                    logging.info(f'Writing warped to .tif: {output_mode_filename}')
                    _ = gdal.Warp(str(output_mode_filename), str(unwarped_tile),
                                                creationOptions=creationOptions)

                    del reduced_stack, reduced_stack_hole_filled
  
            #*MW temp call binary % occurence calculations
            if self.conf.calculate_binary_stats:
                # import pdb; pdb.set_trace()
                logging.info('Calculating binary class frequency outputs')

                binary_calc = 'pct' # 'min', 'median', 'max'
                output_binary_filename = str(output_mode_filename).replace(\
                                       '.mode.tif', f'.class-{binary_calc}.tif')
                # get the classes from config
                composite_classes = list(self.conf.composite_classes.values())
                #*MW TD - should we do anything with the 'tile_grid_ds_bad'? for now just do good or use
                binary_stack = Composite.calculate_binary_class_stats('pct',
                                                    tile_grid_use,
                                                    Path(output_binary_filename),
                                                    overwrite=overwrite,
                                                    nodata=nodata_value,
                                                    # gpu=True, #*MW
                                                    class_values= composite_classes,                
                                                    gpu=True,
                                                    )
                
                if binary_stack is not None: #* if final output tif not yet made
                    binary_stack.rio.write_nodata(nodata_value, inplace=True)
                    unwarped_bin = str(output_binary_filename).replace('.tif', 
                                                                    '-temp.tif')
                    logging.info(f'Writing unwarped to .tif: {str(unwarped_bin)}') 
                    binary_stack.rio.to_raster(str(unwarped_bin), 
                                               dtype = np.uint8, compress='lzw')
                    
                    creationOptions = ['COMPRESS=LZW']
                    logging.info(f'Writing warped to .tif: {output_binary_filename}')
                    _ = gdal.Warp(str(output_binary_filename), 
                            str(unwarped_bin), creationOptions=creationOptions)
                    
                if not self.conf.post_process_combine:
                    del binary_stack # delete iff not combingin
                    
            #*MW - temp approach to calculate nobservations
            output_nobs_filename = str(output_mode_filename).replace('.mode.tif', 
                                                           '.nobservations.tif')
            
            if self.conf.calculate_nobservations and \
                        (overwrite or not Path(output_nobs_filename).exists()):
                
                logging.info('Calculating nobservations')
                
                nobs_array = (tile_grid_ds_good != nodata_value).sum(dim='time')
                nobs_array += (tile_grid_ds_bad != nodata_value).sum(dim='time')

                nobs_array.rio.write_nodata(nodata_value, inplace=True)
                
                unwarped_nobs = output_nobs_filename.replace('.tif', '-temp.tif')
                logging.info(f'Writing unwarped to: {str(unwarped_nobs)}') 
                nobs_array.rio.to_raster(str(unwarped_nobs), dtype = np.uint8,
                                                                compress='lzw')
                logging.info(f'Writing nobservations to .tif: {output_nobs_filename}')
                _ = gdal.Warp(str(output_nobs_filename), str(unwarped_nobs),
                                            creationOptions=['COMPRESS=LZW'])

                del nobs_array

            # Calculate confidence metrics - trend
            if self.conf.calculate_confidence:

                logging.info('Calculating confidence outputs')

                # Remove band since its fixed (size=1)
                lc_array = tile_grid_use.squeeze(dim='band', drop=True)
                lc_array = lc_array.chunk({'time': -1, 'y': 500, 'x': 500})

                if 'trend_confidence' in self.conf.confidence_metrics:
                    logging.info('Calculating trend_confidence outputs')
                    output_filename = str(output_mode_filename).replace(
                                       '.mode.tif', '.trendConfidence.tif')
                    confidence_trend = self._run_accelerated_function(
                        lc_array, calculate_trend_confidence_numba)
                    confidence_trend.rio.to_raster(output_filename)
                    logging.info('Done calculating trend_confidence outputs')
                    logging.info(f'trend_confidence saved at {output_filename}')

                # Calculate confidence metrics - distribution
                if 'distribution_confidence' in self.conf.confidence_metrics:
                    logging.info('Calculating distribution_confidence outputs')
                    output_filename = str(output_mode_filename).replace(
                        '.mode.tif', '.distributionConfidence.tif')
                    confidence_distribution = self._run_accelerated_function(
                        lc_array, calculate_distribution_confidence_numba)
                    confidence_distribution.rio.to_raster(output_filename)
                    logging.info('Done calculating distribution_confidence outputs')
                    logging.info(f'distribution_confidence saved at {output_filename}')

            #*TD - keep here?
            if self.conf.post_process_combine:
                pass
                # create output dir
                # check if output existsif not overwrite
                # check is binary_stack exists
                # if not, caclulate it (see above)
                # call composite.post_process_combine
                # save

            gc.collect()

        return

    def soilMoistureQA(self, tileDF):
        goodSoilMoisture = tileDF['soilM_median'] < 2800
        badSoilMoisture = tileDF['soilM_median'] >= 2800
        goodDF = tileDF[goodSoilMoisture]
        badDF = tileDF[badSoilMoisture]
        badDF = badDF.sort_values(by='soilM_median')
        return goodDF, badDF


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        type=str,
                        required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('-t',
                        '--tiles-filename',
                        dest='tiles_filename',
                        type=str,
                        required=True,
                        help='Filename with tiles to process')
    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'composite'],
                        choices=['preprocess', 'composite'])
    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # setup pipeline object
    pipeline = LandCoverCompositePipeline(args.config)

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess()
    if "composite" in args.pipeline_step:
        pipeline.composite(args.tiles_filename)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    sys.exit()
