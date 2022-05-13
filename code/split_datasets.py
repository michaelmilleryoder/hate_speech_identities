""" Split datasets into with-heg/no-heg and with-control/no-control for comparison """

from collections import Counter
import os
import pandas as pd
import pdb

from tqdm import tqdm


def flexible_sample(df, n):
    """ Sample n rows from df.
        If n <= len(df), sample without replacement.
        If n > len(df), sample all rows and sample additional necessary rows
        (trying to have less duplicates than sampling with replacement)
    """
    if len(df) == 0:
        return df
    
    n_complete_samples = n // len(df)
    remainder = n % len(df)
    
    concat = []
    for _ in range(n_complete_samples):
        concat.append(df)
    concat.append(df.sample(remainder, random_state=9, replace=False))
    return pd.concat(concat).sample(frac=1, random_state=9)


class ComparisonSplits():

    def __init__(self, datasets, removal_groups, hate_ratio):
        """ Args:
                datasets: the datasets to use for constructing the splits
                removal_groups: what sets of identities to remove (hegemonic, an identity category)
        """
        self.datasets = datasets
        self.removal_groups = removal_groups
        self.hate_ratio = hate_ratio
        self.splits = {dataset.name: {'expsplits': {}, 'controlsplits': {}} for dataset in self.datasets}

    def get_stats(self):
        """ Print statistics on dataset splits, including comparisons.
        """
        for dataset_name, s in self.splits.items():
            print(dataset_name)
            for split_type, splits in s.items():
                print(split_type)
                for splitname, split in splits.items():
                    print(splitname)
                    print(f'\tlength: {len(split)}')
                    n_duplicates = split.index.duplicated(keep="first").sum()
                    print(f'\tduplicates: {n_duplicates} ({n_duplicates/len(split):.1%})')
                    hate_proportion = split.hate.value_counts(normalize=True)[True]
                    nonhate_proportion = split.hate.value_counts(normalize=True)[False]
                    print(f'\thate ratio: {hate_proportion:.2f}/{nonhate_proportion:.2f}')
            
                inds_with_special = set(splits['with_special'].index)
                inds_no_special = set(splits['no_special'].index)
                inds_overlap = inds_with_special.intersection(inds_no_special)
                unique_inds_with_special = inds_with_special - inds_overlap
                unique_inds_no_special = inds_no_special - inds_overlap
                split_diffs = {'absolute': len(splits['with_special'][splits['with_special'].index.isin(unique_inds_with_special)])}
                split_diffs['percentage'] = split_diffs['absolute']/len(splits['with_special'])
                print(f"Number of instances different between {list(splits.keys())[0]} and {list(splits.keys())[1]}: {split_diffs['absolute']} ({split_diffs['percentage']:.1%})")
                print()
    
    def sample_to_ratio(self, data, criteria, n_samples=None, n_special=None):
        """ Sample to a specific hate ratio 
            Can also provide a n_samples dictionary with prescribed number of hate and non-hate instances to sample
        """
        # Remove instances by criteria
        special = data.query(criteria)
        no_special = data.loc[~data.index.isin(special.index)]
        
        # Desired sampling of non-hate. Keep all hate rows (for no_special since that's the smallest set)
        n_hate = no_special.hate.sum()
        if n_samples is None:
            n_samples = {
                True: n_hate,
                False: int((n_hate*(1-self.hate_ratio))/self.hate_ratio)
            }
        
        resampled_no_special = no_special.groupby('hate').apply(lambda x: flexible_sample(x, n_samples[x.name]))
        resampled_no_special.index = resampled_no_special.index.droplevel('hate')
        resampled_no_special = resampled_no_special.sample(frac=1, random_state=9)

        # Sample corresponding with_split dataset
        # Want to preserve all the hegemonic/control (special) instances (hate or non-hate) for maximum differences between datasets.
        # So take them out first, then add them back in
        # Want to make this exactly the same as no_split, but with hegemonic/control (special) instances replacing others
        special_hate = special.query('hate')
        special_nonhate = special.query('not hate')
        if n_special is not None:
            n_special_hate, n_special_nonhate = n_special
            special_hate = flexible_sample(special_hate, n_special_hate)
        else:
            n_special_hate = len(special_hate)
            n_special_nonhate = int(len(special_nonhate)/len(data.query('not hate')) * n_samples[False]) # match ratio overall in the dataset
        n_nonhate = len(resampled_no_special.query('not hate'))
        resampled_with_special = pd.concat([
            resampled_no_special.query('hate').sample(n_samples[True]-n_special_hate), 
            special_hate,
            resampled_no_special.query('not hate').sample(n_samples[False] - n_special_nonhate),
            flexible_sample(special_nonhate, n_special_nonhate)
            ], axis=0)
        resampled_with_special = resampled_with_special.sample(frac=1, random_state=9)

        # Test overlap between with_special and no_special to see if it's maximum that it can be
        # Should be exact overlap on not hate that's not special
        unique_heg = Counter(resampled_with_special.index) - Counter(resampled_no_special.index)
        assert sum(unique_heg.values()) == len(special_hate) + n_special_nonhate
        
        return ({'with_special': resampled_with_special,
                'no_special': resampled_no_special},
                n_samples,
                (n_special_hate, n_special_nonhate))

    #def create_heg_control(self):
    #    """ Create heg and control dataset splits """

    #    for dataset in tqdm(self.datasets):
    #        # Heg split
    #        split_criteria = {
    #            'hegsplits': 'group_label == "hegemonic"',
    #            'controlsplits': 'in_control'
    #        }
    #        self.create_splits(split_criteria, dataset)
    #        
    #    # Get stats
    #    self.get_stats()

    #     # Save out
    #    self.save_splits()
    #    print("Saved comparison splits")

    def create_exp_control_splits(self):
        """ Create splits for comparing removing an 'experimental group' (like hegemonic target identities) 
            with a 'control' of random target identities removed 
        """

        exp_criteria_list = []
        for removal_group in self.removal_groups:
            if removal_group == 'hegemonic':
                exp_criteria_list.append('group_label == "hegemonic"')
            else:
                exp_criteria_list.append(f'target_category_{removal_group}')
        exp_criteria = 'and'.join(exp_criteria_list)
        control_criteria = 'and'.join([f'control_{removal_group}' for removal_group in self.removal_groups])
        pdb.set_trace() # check criteria

        for dataset in tqdm(self.datasets):
            split_criteria = {
                'expsplits': exp_criteria,
                'controlsplits': control_criteria
            }
            self.create_splits(split_criteria, dataset)
            
        # Get stats
        self.get_stats()

         # Save out
        self.save_splits()
        print("Saved comparison splits")

    def load_splits(self):
        """ Load experimental and control dataset splits """
        for dataset_name, splits in self.splits.items():
            # Load csv
            base_dirpath = '/storage2/mamille3/data/hate_speech'
            if not os.path.exists(base_dirpath):
                base_dirpath = '/home/mamille3/data/hate_speech'
                assert os.path.exists(base_dirpath)
            dataset_path = os.path.join(base_dirpath, dataset_name, 'processed', "_".join(self.removal_groups)) 
            for splits_name in splits:
                for split_name in ['with_special', 'no_special']:
                    csvpath = os.path.join(dataset_path, f'{dataset_name}_{self.hate_ratio}hate_{splits_name}_{split_name}.csv')
                    self.splits[dataset_name][splits_name][split_name] = pd.read_csv(csvpath, index_col=0)

    def create_splits(self, split_criteria, dataset):
        """ Create splits to compare performance.
            split_criteria: dict with keys of split, values the pandas query to remove instances in the no-split
            Each split should have the same number of instances.
        """

        self.splits[dataset.name] = {}
        
        # Experimental splits
        splits_name = 'expsplits'
        self.splits[dataset.name][splits_name], n_samples, n_special = self.sample_to_ratio(dataset.data, split_criteria[splits_name])
        
        # Control splits
        splits_name = 'controlsplits'
        self.splits[dataset.name][splits_name], n_samples, n_special = self.sample_to_ratio(dataset.data, split_criteria[splits_name], n_samples=n_samples, n_special=n_special)

    def save_splits(self):
        """ Save out splits """
        for dataset_name, splits in self.splits.items():
            # Save out csvs
            base_dirpath = '/storage2/mamille3/data/hate_speech'
            if not os.path.exists(base_dirpath):
                base_dirpath = '/home/mamille3/data/hate_speech'
                assert os.path.exists(base_dirpath)
            dataset_path = os.path.join(base_dirpath, dataset_name, 'processed', "_".join(self.removal_groups)) 
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            for splits_name, s in splits.items():
                for split_name, split in s.items():
                    csvpath = os.path.join(dataset_path, f'{dataset_name}_{self.hate_ratio}hate_{splits_name}_{split_name}.csv')
                    split.to_csv(csvpath)
