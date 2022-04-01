""" Storing and representing attributes of datasets, including processed versions """

import os

import load_data

class Dataset:
    """ Superclass for storing data and attributes of datasets """

    def __init__(self, name, loader=None, dirpath=None, load_paths=None):
        """ Args:
                name: dataset name
                loader: class to use to load the dataset
                load_paths: list of arguments for loading the datasets (like file paths)
        """
        self.name = name
        self.loader = getattr(load_data, f'{self.name.capitalize()}Loader')
        self.dirpath = os.path.join('/storage2/mamille3/data/hate_speech', name)
        self.load_paths = load_paths
        if self.load_paths is None:
            self.load_paths = [f'{name}.csv']
        self.data = None
