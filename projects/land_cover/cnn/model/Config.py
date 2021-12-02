# -*- coding: utf-8 -*-

import os
import yaml
from datetime import datetime
import configparser
import torch
import pandas as pd


# -----------------------------------------------------------------------------
# class Config
# -----------------------------------------------------------------------------
class Config(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, yaml_filename: str, csv_filename: str):

        # setup YAML configuration settings
        self._file_assert(yaml_filename)
        self.yaml_file_path = yaml_filename
        self.read_yaml(yaml_filename)

        # setup CSV data configuration settings
        self._file_assert(csv_filename)
        self.data_file_path = csv_filename
        self.read_csv(csv_filename)

        # set some strict hyperparameter attributes
        self.seed = getattr(self, 'seed', 34)
        self.batch_size = getattr(self, 'batch_size', 16) * \
            getattr(self, 'strategy.num_replicas_in_sync', 1)

        # set some data parameters manually
        self.data_min = getattr(self, 'data_min', 0)
        self.data_max = getattr(self, 'data_max', 10000)
        self.tile_size = getattr(self, 'tile_size', 256)
        self.chunks = {'band': 1, 'x': 2048, 'y': 2048}

        # set some data parameters manually
        self.modify_labels = getattr(self, 'modify_labels', None)
        self.test_size = getattr(self, 'test_size', 0.25)
        self.initial_epoch = getattr(self, 'initial_epoch', 0)
        self.max_epoch = getattr(self, 'max_epoch', 50)
        self.shuffle_train = getattr(self, 'shuffle_train', True)

        # set model parameters
        self.network = getattr(self, 'network', 'unet')
        self.optimizer = getattr(self, 'optimizer', 'Adam')
        self.loss = getattr(self, 'loss', 'categorical_crossentropy')
        self.metrics = getattr(self, 'metrics', ['accuracy'])

        # system performance settings
        self.cuda_devices = getattr(self, 'cuda_devices', '0,1,2,3')
        self.mixed_precission = getattr(self, 'mixed_precission', True)
        self.xla = getattr(self, 'xla', False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # setup directories for input and output
        self.dataset_dir = os.path.join(self.data_output_dir, 'dataset')

        # directories to store new image and labels tiles for training
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.labels_dir = os.path.join(self.dataset_dir, 'labels')

        # logging files
        self.logs_dir = os.path.join(self.data_output_dir, 'logs')
        self.log_file = os.path.join(
            self.logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S") +
            f'-{self.experiment_name}.out')

        # directory to store and retrieve the model object from
        self.model_dir = os.path.join(self.data_output_dir, 'model')

        # directory to store prediction products
        self.predict_dir = os.path.join(self.inference_output_dir)

        # setup directory structure, create directories
        directories_list = [
            self.images_dir, self.labels_dir, self.model_dir,
            self.predict_dir, self.logs_dir]
        self.create_dirs(directories_list)

        # std and means filename for preprocessing and training
        self.std_mean_filename = os.path.join(
            self.dataset_dir, f'{self.experiment_name}_mean_std.npz')

        # new additions
        # self.images_regex = os.path.join(self.dataset_dir, 'images')
        # self.labels_regex = os.path.join(self.dataset_dir, 'labels')

    # -------------------------------------------------------------------------
    # Assert Methods
    # -------------------------------------------------------------------------
    def _file_assert(self, filename: str):
        """
        Assert input file.
        Args:
            filename (str): filename to assert and test
        Return:
            None, exit if assertion encountered
        Example:
            self._file_assert(csv_filename)
            <config_object>._file_assert(csv_filename)
        """
        assert filename, f'Missing fully-qualified path to file {filename}'
        assert os.path.exists(filename), f'{str(filename)} does not exist'

    # -------------------------------------------------------------------------
    # IO Methods - Read
    # -------------------------------------------------------------------------
    def read_yaml(self, filename: str):
        """
        Read YAML configuration file, initialize config attributes.
        Args:
            filename (str): YAML filename to read
        Return:
            <config_object> attributes initialized
        Example:
            self.read_yaml(yaml_filename)
            <config_object>.read_yaml(yaml_filename)
        """
        assert os.path.exists(filename), f'File {filename} not found.'
        try:
            with open(filename) as f:
                hpr = yaml.load(f, Loader=yaml.SafeLoader)
            assert hpr is not None, f'{filename} empty, please add values'
            [setattr(self, key, hpr[key]) for key in hpr]  # init attributes
        except yaml.YAMLError as e:
            raise RuntimeError(f"Syntax Error line {e.problem_mark.line+1}")

    def read_csv(self, filename: str):
        """
        Read CSV configuration file, initialize data_df attribute.
        Args:
            filename (str): CSV filename to read
        Return:
            <config_object>.data_df attribute initialized
        Example:
            self.read_csv(csv_filename)
            <config_object>.read_csv(csv_filename)
        """
        assert os.path.exists(filename), f'File {filename} not found.'
        self.data_df = pd.read_csv(filename)
        assert not self.data_df.isnull().values.any(), f'NaN found: {filename}'

    def read_ini(self, filename: str):
        """
        Read INI configuration file, initialize config attributes.
        Args:
            filename (str): INI filename to read
        Return:
            <config_object> attributes initialized
        Example:
            self.read_ini(ini_filename)
            <config_object>.read_ini(ini_filename)
        """
        assert os.path.exists(filename), f'File {filename} not found.'
        try:  # try initializing config object
            ini = configparser.ConfigParser(
                interpolation=configparser.ExtendedInterpolation())
            ini.read(filename)
            for sect in ini.sections():
                [setattr(self, key, ini[sect][key]) for key in list(ini[sect])]
        except configparser.ParsingError as err:  # abort if incorrect format
            raise RuntimeError(f'Could not parse {err}.')

    # -------------------------------------------------------------------------
    # IO Methods - Create
    # -------------------------------------------------------------------------
    def create_dirs(self, directories: list):
        """
        Create product directories for pipeline execution.
        Args:
            directories (list): list of directories to create
        Return:
            <config_object> directories created
        Example:
            self.read_ini(ini_filename)
            <config_object>.read_ini(ini_filename)
        """
        for d in directories:
            os.makedirs(d, exist_ok=True)
