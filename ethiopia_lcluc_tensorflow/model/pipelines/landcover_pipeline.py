import os
import re
import sys
import time
import logging
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

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

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

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)
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
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF="IF_SAFER",
                    compress=self.conf.prediction_compress,
                    driver=self.conf.prediction_driver,
                    dtype=self.conf.prediction_dtype
                )
                del prediction

                print(Path(output_filename).with_suffix('.tif'))

                """
                # save probability map
                if self.conf.probability_map:

                    probability = xr.DataArray(
                        np.expand_dims(probability[0], axis=-1),
                        name=self.conf.experiment_type,
                        coords=image.coords,
                        dims=image.dims,
                        attrs=image.attrs
                    )

                    # Add metadata to raster attributes
                    probability.attrs['long_name'] = (
                        self.conf.experiment_type)
                    probability.attrs['model_name'] = (
                        self.conf.model_filename)
                    probability = probability.transpose("band", "y", "x")

                    # Set nodata values on mask
                    nodata = probability.rio.nodata
                    probability = probability.where(image != nodata)
                    probability.rio.write_nodata(
                        self.conf.prediction_nodata,
                        encoded=True, inplace=True
                    )

                    # Save output raster file to disk
                    probability.rio.to_raster(
                        output_filename,
                        BIGTIFF="IF_SAFER",
                        compress=self.conf.prediction_compress,
                        driver=self.conf.prediction_driver,
                        dtype='float32'
                    )
                    del probability
                """
                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')
        return

    def validate(self, validation_database: str = None):
        """Perform validation using georeferenced data
        """

        # Gather prediction outputs to validate
        #prediction_filenames = glob(
        #    os.path.join(self.conf.inference_save_dir, '*', '*.6class.tif'))
        #prediction_filenames = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase-ethiopia/6class-global-standardization-256-4band-v2/results/Amhara-MS/*.tif')
        prediction_filenames = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase-ethiopia/ensemble-output-v12lc-v4wm-v1cm/WV02_20100331_M1BS_103001000470DA00-toa.6class-ensemble.tif')
        #prediction_filenames = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase-ethiopia/ensemble-output-v12lc-v4wm-v1cm/WV03_20220204_M1BS_1040010073369100-toa.6class-ensemble.tif')

        #prediction_filenames = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase-ethiopia/6class-global-standardization-256-4band-v12/results/Amhara-MS/WV02_20100331_M1BS_103001000470DA00-toa.6class.tif')
        #prediction_filenames = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase-ethiopia/6class-global-standardization-256-4band-v12/results/Amhara-MS/WV02_20100331_M1BS_103001000470DA00-toa.6class.tif')

        assert len(prediction_filenames) > 0, \
            f'No filenames found under {self.conf.inference_save_dir}'
        logging.info(f'Found {len(prediction_filenames)} to validate.')

        # Open the database to validate from
        if validation_database is not None:
            validation_filename = validation_database
        else:
            if self.conf.validation_database is not None:
                validation_filename = validation_database
            else:
                sys.exit(
                    'You need to specify validation_database in the ' +
                    'configuration file or CLI options.'
                )

        # Read the validation database
        validation_gdf = gpd.read_file(validation_filename)
        logging.info(f'Found {validation_gdf.shape[0]} points to validate.')

        print(validation_gdf.crs)
        print(rasterio.open(prediction_filenames[0]).crs)

        label_column = 'Land_Use'
        crs = 'EPSG:32637'

        # Transfor database to match UTM
        # epoch_df = validation_gdf.to_crs(rasterio.open(prediction_filenames[0]).crs)
        epoch_df = validation_gdf.to_crs('EPSG:32637')

        epoch_df[label_column] = epoch_df[label_column].str.lower()
        epoch_df['class'] = epoch_df[label_column].replace(['crop', 'shrub', 'grass', 'builtup', 'buitlup', 'water'],[0, 1, 2, 3, 3, 4])

        print(epoch_df[label_column].unique())
        print(epoch_df['class'].unique())

        epoch_df['prediction'] = 255
        epoch_df['filename'] = 'filename'

        for filename in prediction_filenames:

            print(Path(filename).stem)

            try:

                src = rasterio.open(filename)
                #print(src.crs)
                epoch_df = epoch_df.to_crs(crs)#src.crs
                #src = src.reproject("EPSG:32628")
                #print(src.crs)
                extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]

                coord_list = [
                    (x, y) for x, y in zip(epoch_df['geometry'].x , epoch_df['geometry'].y)]
                values_composite = [x.item() for x in src.sample(coord_list)]
                #print(values_composite)

                for index, _ in epoch_df.iterrows():
                    if values_composite[index] >= 0 and values_composite[index] < 255:
                        epoch_df.at[index, 'prediction'] = values_composite[index]
                        epoch_df.at[index, 'filename'] = Path(filename).stem
                    #if version == 'training':
                    #    training_shortname = '_'.join((Path(Path(filename).stem).stem).split('_')[:5])
                    #    val_shortname = '_'.join(row['short_filename'].split('_')[:5])
                    #    if values_composite[index] < 5 and training_shortname == val_shortname:
                    #        print(Path(filename).stem, values_composite[index], src.shape)
                    #        epoch_df.at[index, 'Tappan_Training_Data'] = values_composite[index]
                    #        epoch_df.at[index, 'Tappan_Training_Data_filename'] = Path(filename).stem
                    #else:
                    #    if values_composite[index] < 5 and Path(Path(filename).stem).stem == row['short_filename']:
                    #        print(Path(filename).stem, values_composite[index], src.shape)
                    #        epoch_df.at[index, 'composite'] = values_composite[index]
                    #        epoch_df.at[index, 'composite_filename'] = Path(filename).stem

            except:
                continue

        #print(epoch_df.composite.unique())
        epoch_df.to_file('/home/jacaraba/woubet-2010.gpkg', format='GPKG')

        epoch_df = epoch_df[epoch_df['prediction'] != 255]

        print(epoch_df)

        accuracy = accuracy_score(epoch_df['class'], epoch_df['prediction'], normalize=True, sample_weight=None)
        print("Accuracy", accuracy)

        balanced_accuracy = balanced_accuracy_score(epoch_df['class'], epoch_df['prediction'], sample_weight=None)
        print("Balance Accuracy", accuracy)

        print(epoch_df['class'].value_counts())

        confusion_matrix_lc = confusion_matrix(epoch_df['class'], epoch_df['prediction'], normalize='true')
        print(confusion_matrix_lc)

        confusion_matrix_lc = confusion_matrix(epoch_df['class'], epoch_df['prediction'])
        print(confusion_matrix_lc)

        print(epoch_df.shape)

        return 


    def test(self):
        """
        import os
        import glob
        import warnings
        import itertools
        import xarray as xr
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf
        import matplotlib.colors as pltc
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.metrics import classification_report
        import csv
        import re
        ​
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ​
        def confusion_matrix_func(y_true=[], y_pred=[], nclasses=4, norm=True):
            
            Args:
                y_true:   2D numpy array with ground truth
                y_pred:   2D numpy array with predictions (already processed)
                nclasses: number of classes
            Returns:
                numpy array with confusion matrix
            
        ​
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
        ​
            label_name = np.unique(y_pred)
        ​
            # get overall weighted accuracy
            accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
        ​
            # print(classification_report(y_true, y_pred))
            if len(label_name) != 4:
                target_names = ['other-vegetation','tree','cropland']
            else:
                target_names = ['other-vegetation','tree','cropland','burned']
            report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        ​
            tree_recall = report['tree']['recall']
            crop_recall = report['cropland']['recall']
        ​
            tree_precision = report['tree']['precision']
            crop_precision = report['cropland']['precision']
        ​
            ## get confusion matrix
            con_mat = tf.math.confusion_matrix(
                labels=y_true, predictions=y_pred, num_classes=nclasses
            ).numpy()
        ​
            # print(con_mat.sum(axis=1)[:, np.newaxis])
            # print(con_mat.sum(axis=1)[:, np.newaxis][0])
            # weights = [con_mat.sum(axis=1)[:, np.newaxis][0][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][1][0]/(5000*5000),
            # con_mat.sum(axis=1)[:, np.newaxis][2][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][3][0]/(5000*5000)]
        ​
            # print(weights)
            # get overall weighted accuracy
            # accuracy = accuracy_score(y_true, y_pred, normalize=False, sample_weight=weights)
            # print(con_mat)
        ​
            if norm:
                con_mat = np.around(
                    con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
                    decimals=2
                )
        ​
            # print(con_mat)
        ​
            # print(con_mat.sum(axis=1)[:, np.newaxis])
            where_are_NaNs = np.isnan(con_mat)
            con_mat[where_are_NaNs] = 0
            return con_mat, accuracy, balanced_accuracy, tree_recall, crop_recall, tree_precision, crop_precision
        ​
        ​
        def plot_confusion_matrix(cm, label_name, model, class_names=['a', 'b', 'c']):
            
            Returns a matplotlib figure containing the plotted confusion matrix.
            Args:
                cm (array, shape = [n, n]): a confusion matrix of integer classes
                class_names: list with classes for confusion matrix
            Return: confusion matrix figure.
            
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            # plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            # Use white text if squares are dark; otherwise black.
            threshold = 0.55  # cm.max() / 2.
            # print(cm.shape[0], cm.shape[1]) #, threshold[0])
        ​
            print(label_name[:-27])
        ​
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(f'/home/geoint/tri/nasa_senegal/confusion_matrix/cas-tl-wcas/{model}-{label_name[:-27]}_cfn_matrix_4class.png')
            # plt.show()
            plt.close()
        ​
        ​
        # Press the green button in the gutter to run the script.
        if __name__ == '__main__':
        ​
            classes = ['Other', 'Tree/shrub', 'Croplands', 'Burned Area']  # 6-Cloud not present
            colors = ['brown', 'forestgreen', 'orange', 'grey']
            colormap = pltc.ListedColormap(colors)
        ​
            labels = sorted(glob.glob('/home/geoint/tri/allwcasmasks/*.tif'))
        ​
            out_dir = '/home/geoint/tri/nasa_senegal/pred_eval/increase_wCAS_set1/'
        ​
            with open(f'{out_dir}66-0.12-cas15-tl-wcas-8ts-set2-0525_stat_results.csv', 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(['tappan', 'overall-accuracy', 'accuracy', 'tree-recall', 'crop-recall', 'tree-precision', 'crop-precision'])
        ​
                for lf in labels:
        ​
                    print(os.path.basename(lf))
                    file_name = os.path.basename(lf)
        ​
                    name = file_name[:-9]
        ​
                    search_term_lf = re.search(r'/allwcasmasks/(.*?)_mask.tif', lf).group(1)
                    # search_term_lf = re.search(r'/CAS_West_masks/(.*?)_mask', lf).group(1)
                    print(search_term_lf)
        ​
                    pf = '/home/geoint/tri/nasa_senegal/predictions/66-0.12-cas15-tl-wcas-8ts-set2-0525/images/' + search_term_lf + '_data.landcover.tif'
        ​
                    search_term_pf = re.search(r'predictions.(.*?)/images', pf).group(1)
                    print(search_term_pf)
        ​
                    # open filenames
                    label = np.squeeze(xr.open_rasterio(lf).values)
                    prediction = np.squeeze(xr.open_rasterio(pf).values)
        ​
                    ## group label pixel to 4 classes
                    # label[label == 5] = 3  # merge burned area to other vegetation
                    label[label == 7] = 3  # merge no-data area to shadow/water
                    label[label == 4] = 3
                    label[label == 3] = 0
                    label[label == 5] = 3
        ​
                    # some preprocessing
                    prediction[prediction == -10001] = 0
                    prediction[prediction == 255] = 0
                    # prediction[prediction == 3] = 0
        ​
                    print("Unique pixel values of prediction: ", np.unique(prediction))
                    print("Unique pixel values of label: ", np.unique(label))
        ​
                    cnf_matrix, accuracy, balanced_accuracy, tree_recall, crop_recall, tree_precision, crop_precision = confusion_matrix_func(y_true=label, y_pred=prediction, nclasses=len(classes), norm=True)
        ​
                    writer.writerow([name, accuracy, balanced_accuracy, tree_recall, crop_recall, tree_precision, crop_precision])
        
                    print("Overall Accuracy: ", accuracy)
                    print("Balanced Accuracy: ", balanced_accuracy)
    
                    plot_confusion_matrix(cnf_matrix, file_name, search_term_pf, class_names=classes)
        """
        return
