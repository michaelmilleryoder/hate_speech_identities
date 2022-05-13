""" Load, process multiple datasets """

import pdb

import numpy as np
import pandas as pd

from load_process_dataset import DataLoader, get_dummy_columns

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
        """ Get, save out terms with similar frequencies across datasets for comparison
            to identities that are removed for performance comparisons
        """

        eg_data = self.datasets[0].data
        removal_groups = ['hegemonic'] + [col[len('target_category_'):] for col in eg_data.columns if col.startswith('target_category_')]
        
        # Get frequncies of normalized labels across datasets
        group_targets = [] # target_group, group_label, dataset, count
        for dataset, loader in zip(self.datasets, self.loaders):
            target_counts = dataset.target_counts()
            target_counts['dataset'] = dataset.name
            target_counts['group_label'] = target_counts.group.map(lambda group: loader.group_labels.get(group, 'other'))
            target_counts['identity_categories'] = target_counts.group.map(lambda group: loader.assign_categories([group]))
            target_counts = get_dummy_columns(target_counts, 'identity_categories', exclude=['other'])
            group_targets.append(target_counts)
        target_dataset_counts = pd.concat(group_targets)
        # Fill NaNs with False
        identity_cols = [col for col in target_dataset_counts.columns if col.startswith('identity_category')]
        target_dataset_counts[identity_cols] = target_dataset_counts[identity_cols].fillna(False).astype(bool)
        #target_dataset_counts.drop_duplicates(subset=['group', 'dataset'], inplace=True)

        for removal_group in removal_groups:
            # used to restrict hegemonic control terms to just be marginalized (instead of any other)
            if removal_group == 'hegemonic':
                criteria = 'group_label == "hegemonic"'
                removal_col = 'group_label'
            else:
                criteria = f'identity_category_{removal_group}'
                removal_col = f'identity_categories'

            ## Get distributions of counts over datasets for normalized labels
            removal_targets = target_dataset_counts.query(criteria)
            removal_counts = removal_targets.drop(columns=[removal_col] + [col for col in removal_targets.columns if 'identity_category' in col])
            removal_counts = removal_counts.pivot_table(index=['group'], columns=['dataset'])
            # Fill in datasets in which terms may not appear
            missing_datasets = [dataset.name for dataset in self.datasets if not dataset.name in removal_counts.columns.get_level_values('dataset')]
            for dataset in missing_datasets:
                removal_counts[('count', dataset)] = None
            removal_counts.fillna(0, inplace=True)
            log_removal_counts = removal_counts.apply(np.log2).replace(-np.inf, -1)
            log_removal_counts['magnitude'] = np.linalg.norm(
                    log_removal_counts[[col for col in log_removal_counts.columns if col[0] == 'count']], axis=1)
            log_removal_counts = log_removal_counts.sort_values('magnitude', ascending=False).drop(columns='magnitude')
            
            # Find non-removal terms with similar frequency distributions across datasets as removal ones
            nonremoval_targets = target_dataset_counts.query('not ' + criteria)
            nonremoval_counts = nonremoval_targets.drop(columns=[removal_col]+ [col for col in removal_targets.columns if 'identity_category' in col])
            nonremoval_counts = nonremoval_counts.pivot_table(index=['group'], columns=['dataset'])
            nonremoval_counts.fillna(0, inplace=True)
            log_nonremoval_counts = nonremoval_counts.apply(np.log2).replace(-np.inf, -1)
            nonremoval = log_nonremoval_counts.copy()
            control_terms = []
            for removal_term, removal_vec in log_removal_counts.iterrows():
                distances = np.linalg.norm(nonremoval.values - removal_vec.values, axis=1)
                closest_nonremoval = nonremoval.index[np.argmin(distances)]
                control_terms.append(closest_nonremoval)
                nonremoval.drop(closest_nonremoval, inplace=True) 

            # Save control terms out
            outpath = f'../resources/control_identity_terms_{removal_group}.txt'
            print(f"Control terms for {removal_group}:")
            with open(outpath, 'w') as f:
                for term in control_terms:
                    print(f'\t{term}')
                    f.write(f'{term}\n')
        
            # Check counts across datasets for heg and control
            #print("Removal counts compared with control counts")
            #print(removal_counts.sum())
            #print(removal_counts.sum().sum())
            #print()
            #print(nonremoval_counts.loc[control_terms].sum())
            #print(nonremoval_counts.loc[control_terms].sum().sum())
