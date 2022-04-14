""" Load, process multiple datasets """

import pdb

import numpy as np
import pandas as pd

from load_process_dataset import DataLoader

class DatasetsLoader:
    """ Load and/or process multiple datasets 
    """

    def __init__(self, datasets):
        """ Args:
                datasets: datasets to load
        """
        self.datasets = datasets
        self.loaders = None

    def load_datasets(self, reprocess=False):
        """ Load and process raw datasets or load processed datasets """
        if reprocess:
            self.load_process_datasets()
        else:
            self.load_processed_datasets()

    def load_processed_datasets(self):
        """ Load processed datasets """
        self.loaders = []
        for dataset in self.datasets:
            loader = dataset.loader()
            loader.load_processed(dataset) 

    def load_process_datasets(self):
        """ Load and process datasets """
        self.load_raw_datasets()
        self.process_datasets()
        self.get_control_terms()
        self.label_control()
        self.save_datasets()
        
    def load_raw_datasets(self):
        """ Load datasets """
        self.loaders = []
        for dataset in self.datasets:
            loader = dataset.loader() 
            loader.load(dataset)
            self.loaders.append(loader)

    def process_datasets(self):
        """ Process datasets """
        for dataset, loader in zip(self.datasets, self.loaders):
            loader.process(dataset)
    
    def label_control(self):
        """ Label control instances in datasets. This is separate from processing since
            calculating control terms depends on other processing. """
        for dataset, loader in zip(self.datasets, self.loaders):
            loader.label_control(dataset)

    def save_datasets(self):
        """ Save processsed datasets """
        for dataset, loader in zip(self.datasets, self.loaders):
            loader.save(dataset)

    def get_control_terms(self):
        """ Get, save out marginalized terms with similar frequencies across datasets for comparison
            to hegemonic terms
        """
        
        # Get frequncies of normalized labels across datasets
        group_targets = [] # target_group, group_label, dataset, count
        for dataset, loader in zip(self.datasets, self.loaders):
            target_counts = dataset.target_counts()
            target_counts['dataset'] = dataset.name
            target_counts['group_label'] = target_counts.group.map(lambda group: loader.group_labels.get(group, 'other'))
            group_targets.append(target_counts)
        target_dataset_counts = pd.concat(group_targets)
        target_dataset_counts.drop_duplicates(inplace=True)

        ## Get distributions of counts over datasets for normalized hegemonic labels
        heg_targets = target_dataset_counts.query('group_label == "hegemonic"')
        heg_counts = heg_targets.drop(columns=['group_label']).pivot_table(index=['group'], columns=['dataset'])
        heg_counts.fillna(0, inplace=True)
        log_heg_counts = heg_counts.apply(np.log2).replace(-np.inf, -1)
        log_heg_counts['magnitude'] = np.linalg.norm(log_heg_counts[[col for col in log_heg_counts.columns if col[0] == 'count']], axis=1)
        log_heg_counts = log_heg_counts.sort_values('magnitude', ascending=False).drop(columns='magnitude')
        
        # Find marginalized terms with similar frequency distributions across datasets as margemonic ones
        marg_targets = target_dataset_counts.query('group_label == "marginalized"')
        marg_counts = marg_targets.drop(columns=['group_label']).pivot_table(index=['group'], columns=['dataset'])
        marg_counts.fillna(0, inplace=True)
        #log_marg_counts = marg_counts.apply(np.log2).replace(-np.inf, -1)
        log_marg_counts = marg_counts.apply(np.log2).replace(-np.inf, -1)
        marg = log_marg_counts.copy()
        control_terms = []
        for heg_term, heg_vec in log_heg_counts.iterrows():
            distances = np.linalg.norm(marg.values - heg_vec.values, axis=1)
            closest_marg = marg.index[np.argmin(distances)]
            control_terms.append(closest_marg)
            marg.drop(closest_marg, inplace=True) 

        # Save control terms out
        outpath = '../resources/control_identity_terms.txt'
        print("Control terms:")
        with open(outpath, 'w') as f:
            for term in control_terms:
                print(f'\t{term}')
                f.write(f'{term}\n')
        
        # Check counts across datasets for heg and control
        print("Heg counts compared with control counts")
        print(heg_counts.sum())
        print(heg_counts.sum().sum())
        print()
        print(marg_counts.loc[control_terms].sum())
        print(marg_counts.loc[control_terms].sum().sum())
