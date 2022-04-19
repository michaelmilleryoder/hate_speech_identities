""" Storing and representing attributes of datasets, including processed versions """

import os
from collections import Counter
import pdb

import pandas as pd

import load_process_dataset

class Dataset:
    """ Superclass for storing data and attributes of datasets """

    def __init__(self, name, loader=None, dirpath=None, load_paths=None):
        """ Args:
                name: dataset name
                loader: class to use to load the dataset
                load_paths: list of arguments for loading the datasets (like file paths)
        """
        self.name = name
        self.loader = getattr(load_process_dataset, f'{self.name.capitalize()}Loader')
        if os.path.exists('/storage2/mamille3/data/hate_speech'): # should probably be an arg instead
            self.dirpath = os.path.join('/storage2/mamille3/data/hate_speech', name)
        else:
            self.dirpath = os.path.join('/usr0/home/mamille3/data/hate_speech', name)
        self.load_paths = load_paths
        if self.load_paths is None or len(self.load_paths)==0:
            self.load_paths = [f'{name}.csv']
        self.data = None

    def target_counts(self, just_hate=False):
        """ Returns a series of counts of normalized group targets (from target_groups col) 
            Args:
                just_hate: if True, just return targets from instances marked for hate
        """
        if just_hate:
            data = self.data.query('hate')
        else:
            data = self.data
        targets_flattened = [t for targets in data.target_groups.dropna() for t in targets]
        targets = Counter(targets_flattened)
        target_counts = pd.DataFrame(pd.Series(targets), columns=['count']).rename_axis('group').reset_index()
        target_counts.sort_values(['count'], ascending=False, inplace=True)
        return target_counts
