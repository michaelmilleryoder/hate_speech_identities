""" Create datasets by identity targets """

import json
import os
import pickle
import pdb        
from collections import Counter

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
tqdm.pandas()

from split_datasets import flexible_sample


def get_stats(data, identity):
    """ Print statistics on dataset
    """
    
    print(f'\tlength: {len(data)}')
    n_duplicates = data.index.duplicated(keep="first").sum()
    print(f'\tduplicates: {n_duplicates} ({n_duplicates/len(data):.1%})')
    hate_proportion = data.hate.value_counts(normalize=True)[True]
    nonhate_proportion = data.hate.value_counts(normalize=True)[False]
    print(f'\thate ratio: {hate_proportion:.2f}/{nonhate_proportion:.2f}')
    with_identity_group = data.query('identity_group == @identity')
    identity_hate = with_identity_group.query('hate')
    identity_nonhate = with_identity_group.query('not hate')
    print(f'\tNumber of instances with {identity}: {len(with_identity_group)} ({len(with_identity_group)/len(data):.1%})')
    print(f'\t\tlabeled hate: {len(identity_hate)}')
    print(f'\t\tlabeled nonhate: {len(identity_nonhate)}')
    print()
        

class IdentityDatasetCreator:
    
    def __init__(self, processed_datasets, hate_ratio):
        """ Args:
                processed_datasets: full datasets to split by identity
        """ 
        self.processed_datasets = processed_datasets
        self.hate_ratio = hate_ratio
        self.identity_groups = None
        self.grouped_identities = None
        self.selected_groups = None
        self.expanded_datasets = None
        self.folds = {}

    def create_datasets(self):
        """ Create and return identity datasets """

        # Count, select which identities to group in datasets
        threshold = 500
        self.dataset_identity_groups(threshold) # can also load them if they are calculated

        # Expand datasets with annotations for identity groups 
        # (duplicate instances with multiple target identity groups)
        print("Expanding datasets with target identity groups...")
        self.expand_datasets()

        # Create identity datasets
        self.form_identity_datasets()

        return self.folds

    def form_identity_datasets(self):
        """ For each selected identity target in selected datasets, form datasets with training and test folds """

        for dataset_name in tqdm(sorted(set([dataset_name for dataset_name, identity in self.selected_groups]))):
            identity_datasets = self.create_identity_datasets(dataset_name, self.expanded_datasets[dataset_name])
            
            # Split into train and test folds
            self.folds.update(self.create_folds(identity_datasets))
            print('*********************')
            
        # Save out
        self.save_identity_datasets()
        print("Saved splits")

    def create_folds(self, identity_datasets):
        """ Create train and test folds
        """
        
        folds = {}
        for name, data in identity_datasets.items():
            print(name)
            # Split into train/test 60/40
            # Get, print differences between with-heg vs no-heg splits (can use indexes)
            inds = {}
            folds[name] = {}

            # Splitting unique indices so even if nonhate in datasets were oversampled, there are no duplicates across folds
            inds['train'], inds['test'] = train_test_split(data.index.unique(), test_size=0.4)

            for fold in ['train', 'test']:
                folds[name][fold] = data[data.index.isin(inds[fold])]
                # Simple stats
                print(f'\t{fold} length: {len(folds[name][fold])} ({folds[name][fold].hate.value_counts(normalize=True)[True]:.1%} hate)')
        
        return folds

    def save_identity_datasets(self):
        # Save out folds
        outpath = f'/storage2/mamille3/hegemonic_hate/data/identity_splits_{self.hate_ratio}hate.pkl'
        with open(outpath, 'wb') as f:
            pickle.dump(self.folds, f)
        print("Saved identity folds out")
            
        # Save out csv
        # dataset_path = f'/storage2/mamille3/data/hate_speech/{dataset}/processed'
        # if not os.path.exists(dataset_path):
        #     os.makedirs(dataset_path)
        # for splits_name, s in splits.items():
        #     for split_name, split in s.items():
        #         csvpath = os.path.join(dataset_path, f'{dataset}_{hate_ratio}hate_{split_name}.csv')
        #         split.to_csv(csvpath)
        
    def sample_to_ratio(self, dataset, data, identity):
        """ Sample to a specific hate ratio 
            Just select instances with that identity group for hate.
            Negative examples either should have no target specified or just that identity group
        """
        # Find instances targeting that identity group
        identity_data = data.query('identity_group == @identity')
        
        # Desired sampling of non-hate. Keep all hate rows (for no_special since that's the smallest set)
        # Wouldn't need this anymore
        n_hate = identity_data.hate.sum()
        n_samples = {
            True: n_hate,
            False: int((n_hate*(1-self.hate_ratio))/self.hate_ratio)
        }
        
        identity_hate = identity_data.query('hate')
        identity_nonhate = pd.concat([identity_data.query('not hate'), 
            data.loc[data.hate==False & data.target_groups.map(lambda x: isinstance(x, float) or len(x)==0)]
                                     ])
        resampled = pd.concat([
            identity_hate,
            flexible_sample(identity_nonhate, n_samples[False])
            ], axis=0)
        resampled = resampled.sample(frac=1, random_state=9)

        if len(resampled) == 0:
            pdb.set_trace()
        get_stats(resampled, identity)
        return resampled

    def create_identity_datasets(self, dataset, data):
        """ Create splits to compare performance.
            split_criteria: dict with keys of split, values the pandas query to remove instances in the no-split
            Each split should have the same number of instances.
            Args:
                data: expanded dataset
        """
        
        identity_datasets = {}
        dataset_identities_selected = [(dataset_name, identity) for dataset_name, identity in self.selected_groups if dataset_name==dataset]
        for dataset_name, identity in dataset_identities_selected:
            identity_datasets[(dataset, identity)] = self.sample_to_ratio(dataset, data, identity)
        return identity_datasets

    def assign_groups(self, identities):
        if not isinstance(identities, list):
            return []
        groups = [self.identity_groups.get(identity, [identity] if identity in self.grouped_identities else []) for identity in identities]
        flattened = [group for identities in groups for group in identities]
        return flattened

    def expand_datasets(self):
        """ Expand datasets with annotations for identity groups 
            (duplicate instances with multiple target identity groups)
        """
        # Select instances based on selected datasets and identity groups

        self.expanded_datasets = {}
        for dataset in tqdm(self.processed_datasets):
            print(dataset.name)
            data = dataset.data.copy()
            data['identity_groups'] = data['target_groups'].map(self.assign_groups)
            #s = data['identity_groups'].progress_apply(pd.Series, 1).stack() # takes forever
            #s.index = s.index.droplevel(-1)
            #s.name = 'identity_group'
            #del data['identity_groups']
            #self.expanded_datasets[dataset.name] = data.join(s)
            self.expanded_datasets[dataset.name] = data.explode('identity_groups').rename(columns={'identity_groups': 'identity_group'})

        # Save out since takes such a long time
        # Haven't written a loader for this since once I have the identity dataset won't need to run this
        outpath = '/storage2/mamille3/hegemonic_hate/tmp/expanded_datasets.pkl'
        with open(outpath, 'wb') as f:
            pickle.dump(self.expanded_datasets, f)

    def load_identity_groups(self):
        """ Loads identity groupings into self.identity_groups """
        path = '/storage2/mamille3/hegemonic_hate/resources/identity_groups.json'
        
        with open(path, 'r') as f:
            self.identity_groups = json.load(f)

        self.grouped_identities = set(self.identity_groups.keys()).union(
            set([val for vals in self.identity_groups.values() for val in vals]))

    def dataset_identity_groups(self, threshold):
        """ Return a list of tuples (dataset, identity) of identity targets with a
            minimum number of instances of hate against them in that dataset """

        self.load_identity_groups()
        
        # Get frequncies of grouped labels across datasets
        group_targets = [] # target_group, group_label, dataset, count
        for dataset in self.processed_datasets:
            #if dataset.name == 'cad':
            #    pdb.set_trace()
            target_counts = dataset.target_counts(just_hate=True)
            target_counts['dataset'] = dataset.name
            target_counts['identity_group'] = target_counts.group.map(lambda x: self.identity_groups.get(x, [x] if x in self.grouped_identities else []))
            target_counts = target_counts.loc[target_counts.identity_group.map(lambda x: len(x) > 0)]
            group_targets.append(target_counts)
        target_dataset_counts = pd.concat(group_targets)
        target_dataset_counts.drop_duplicates(subset=['group', 'dataset'], inplace=True)

        target_group_counts = target_dataset_counts.explode('identity_group')
        dataset_group_counts = target_group_counts.groupby(['dataset', 'identity_group'])['count'].sum()
        filtered = dataset_group_counts[dataset_group_counts >= threshold]
        
        # Save out
        selected_dataset_groups = filtered.index.tolist()
        outpath = '/storage2/mamille3/hegemonic_hate/tmp/selected_dataset_groups.pkl'
        with open(outpath, 'wb') as f:
            pickle.dump(selected_dataset_groups, f)

        self.selected_groups = selected_dataset_groups
