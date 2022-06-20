#!/usr/bin/env python
# coding: utf-8

"""
# Load, process datasets.  
# Label instances within datasets with hegemonic/marginalized/other target group labels
# Build control dataset splits taking out random marginalized identities to compare with taking out hegemonic identities.
"""

import json
import os
import pdb
import io
from contextlib import redirect_stdout, redirect_stderr
from ast import literal_eval

import numpy as np
import pandas as pd
import datasets # from HuggingFace
from tqdm import tqdm
tqdm.pandas()
from sklearn.preprocessing import MultiLabelBinarizer


def dummy_colname(split_colname, value):
    """ Transform and combine a column name and value to a good column name 
        Used by get_dummy_columns
        Args:
            split_colname: the name of the original column which is being split into indicator columns
            value: the value, to be a suffix of the new indicator column
    """
    name_transform = {'target_categories': 'target_category', 
                        'identity_categories': 'identity_category',
                     }
    # Would be more efficient not to define this dict every time probably

    return f'{name_transform.get(split_colname, split_colname)}_{value.replace("/", "_")}'


def get_dummy_columns(data, colname, exclude=None, include=None):
    """ Create boolean columns for each value in a column 
        Args:
            data: the dataframe to manipulate
            colname: the name of the list column whose values will become boolean columns
            exclude: values to exclude
            include: values to include, even if they are not present in the column being split
        Returns new dataframe with the columns
    """
    mlb = MultiLabelBinarizer()
    dummy_cols = pd.DataFrame(mlb.fit_transform(data[colname]), 
            columns=[dummy_colname(colname, val) for val in mlb.classes_],
            index=data.index).astype(bool)
    if exclude is not None:
        dummy_cols.drop(columns=[dummy_colname(colname, exclude_val) for exclude_val in exclude], 
                errors='ignore', inplace=True)
    if include is not None:
        include_vals = [include_val for include_val in include if dummy_colname(colname, include_val) not in dummy_cols.columns]
        dummy_cols = dummy_cols.assign(**{dummy_colname(colname, include_val): False for include_val in include_vals})
    data = pd.concat([data, dummy_cols], axis=1)
    return data


class DataLoader:
    """ Load, process a dataset.
        TODO: probably make this DataProcessor instead, consider moving basic saving and loading to data.py
    """

    def __init__(self):
        """ Constructor """
        self.groups_norm = None
        self.group_labels = None
        self.identity_categories = None
        self.load_resources()

    def load_resources(self):
        """ Load resources for labeling """
        # Load group name normalization dict
        group_norm_path = '../resources/normalized_groups.json'
        with open(group_norm_path, 'r') as f:
            self.groups_norm = json.load(f) 

        # Load group labels dict
        group_label_path = '../resources/group_labels.json'
        with open(group_label_path, 'r') as f:
            self.group_labels = json.load(f) 

        # Load group categories dict
        identity_categories_path = '../resources/identity_categories.json'
        with open(identity_categories_path, 'r') as f:
            self.identity_categories = json.load(f) 

        # Make sure all normalized terms have group labels
        norm_terms_no_label = [label for label in set(self.groups_norm.values()) if label not in self.group_labels and label != '']
        assert len(norm_terms_no_label) == 0

    def assign_label(self, targets):
        """ Assign labels to target groups """
        if targets is None or isinstance(targets, float) or len(targets) == 0:
            label = None
        else:
            label = 'other'
            labels = {self.group_labels.get(target, 'other') for target in targets}
            if 'marginalized' in labels and not 'hegemonic' in labels:
                label = 'marginalized'
            elif 'hegemonic' in labels:
                label = 'hegemonic'
        return label

    def assign_categories(self, targets):
        """ Assign identity categories to target groups """
        if targets is None or isinstance(targets, float) or len(targets) == 0:
            flattened = []
        else:
            categories = [self.identity_categories.get(target, ['other']) for target in targets]
            flattened = sorted(set([category for category_list in categories for category in category_list]))
        return flattened

    def load(self, dataset):
        """ Load a dataset to dataset.data. Usually overwritten by subclasses
        """
        print(f"\tLoading {dataset.name}...")
        dataset.data = pd.read_csv(os.path.join(dataset.dirpath, dataset.load_paths[0]), index_col=0)

    def process(self, dataset):
        """ Do processing and naming of columns shared across datasets
         Each dataset dataframe will have a
         * index named 'text_id' with unique post/comment ID,
         * 'hate' binary hate_speech column, 
         * 'target_groups' list column of normalized identities
         * 'group_label' column {hegemonic, marginalized, other}
         * 'text' column
            Result will be saved in dataset.data
        """

        print(f'Processing {dataset.name}...')
    
        # Assign binary hate column
        self.label_hate(dataset)

        # Get target_groups list column
        self.extract_target_groups(dataset)

        # Rename text col
        self.rename_text_column(dataset)

        # Assign group label to instances
        dataset.data['group_label'] = dataset.data.target_groups.map(self.assign_label)

        # Assign group categories to instances
        dataset.data['target_categories'] = dataset.data.target_groups.map(self.assign_categories)
        dataset.data = get_dummy_columns(dataset.data, 'target_categories', 
                exclude=['other'], include=['race_ethnicity', 'religion', 'gender', 'sexuality'])

        # Drop nans in text column
        dataset.data = dataset.data.dropna(subset=['text'], how='any')

        # Check, rename index
        dataset.data.index.name = 'text_id'
        assert not dataset.data.index.duplicated(keep=False).any()
        #if dataset.data.index.duplicated(keep=False).any():
        #    pdb.set_trace()

    def label_control(self, dataset):
        """ Label which instances target terms in the control set for different removal groups """
        ""

        control_paths = [os.path.join('../resources', name) for name in os.listdir('../resources') if name.startswith('control_identity_terms')]

        # Load control group terms
        for path in control_paths:
            with open(path, 'r') as f:
                control_terms = f.read().splitlines()
            removal_group = path[len('../resources/control_identity_terms_'):].split('.')[0]

            # Control group column
            dataset.data[f'control_{removal_group}'] = dataset.data.target_groups.map(lambda targets: any(t in control_terms for t in targets) if isinstance(targets, list) else False)

    def extract_target_groups(self, dataset):
        """ Extract target groups into a list column. Usually overwritten by specific datasets. """
        assert 'target_groups' in dataset.data.columns

    def label_hate(self, dataset):
        """ Assign a binary hate column. Usually overwritten for specific datasets. """
        assert 'hate' in dataset.data.columns and dataset.data['hate'].dtype == bool

    def rename_text_column(self, dataset):
        """ Rename text column to 'text'. Usually overwritten for specific datasets. """
        assert 'text' in dataset.data.columns

    def save(self, dataset):
        """ Save out dataset """
        out_dirpath = os.path.join(dataset.dirpath, 'processed')
        if not os.path.exists(out_dirpath):
            os.mkdir(out_dirpath)
        outpath = os.path.join(out_dirpath, f'{dataset.name}_binary_hate_targets.csv')
        dataset.data.to_csv(outpath)
        print(f"\tSaved out processed {dataset.name}")

    def load_processed(self, dataset):
        """ Load processed dataset """
        print(f"Loading processed {dataset.name}")
        dirpath = os.path.join(dataset.dirpath, 'processed')
        path = os.path.join(dirpath, f'{dataset.name}_binary_hate_targets.csv')
        #dataset.data = pd.read_csv(path, index_col=0, converters={'target_groups': literal_eval})
        dataset.data = pd.read_csv(path, index_col=0, low_memory=False)
        dataset.data['target_groups'] = dataset.data.target_groups.map(literal_eval)
        dataset.data['target_categories'] = dataset.data.target_categories.map(literal_eval)


class Kennedy2020Loader(DataLoader):
    """ Load Kennedy+2020 dataset """

    def load(self, dataset):
        """ Load/download data """
        print("\tLoading kennedy2020...")
        with redirect_stderr(io.StringIO()) as f:
            data = datasets.load_dataset(*dataset.load_paths)['train'].to_pandas()

        # Combine annotations for the same comments
        # Just do an "any" approach for targets, mean for hate speech scores
        target_cols = [col for col in data.columns.tolist() if col.startswith('target_')]
        aggregators = {col: 'any' for col in target_cols}
        aggregators.update({'hate_speech_score': 'mean', 'text': 'first'})
        comments = data.groupby('comment_id').agg(aggregators)
        dataset.data = comments

    def label_hate(self, dataset):
        threshold = 1
        dataset.data['hate'] = dataset.data['hate_speech_score'].map(lambda x: True if x>threshold else False)

    def extract_target_groups(self, dataset):
        target_cols = [col for col in dataset.data.columns.tolist() if col.startswith('target_')]
        group_target_cols = [col for col in target_cols if 'disability' in col or (col.count('_')>1 and 'other' not in col)]
        dataset.data['target_groups'] = dataset.data[group_target_cols].apply(self.extract_targets, axis=1)
        
    @classmethod
    def extract_group(cls, colname: str):
        # Extract group name from column name
        if 'disability' in colname:
            group = '_'.join(colname.split('_')[1:])
        else:
            group = '_'.join(colname.split('_')[2:])
        return group
    
    def extract_targets(self, row):
        """ Args:
                row: row as a Series (from apply)
        """
        targets = [self.groups_norm.get(self.extract_group(colname), self.extract_group(colname)) \
            for colname in row[row==True].index if self.groups_norm.get(
            self.extract_group(colname), self.extract_group(colname)) != '']
        if len(targets) > 0:
            return targets
        else:
            return []
        
        
class CadLoader(DataLoader):
    """ Contextual Abuse Dataset (CAD) """

    def load(self, dataset):
        print("\tLoading cad...")
        csvpath = os.path.join(dataset.dirpath, dataset.load_paths[0])
        dataset.data = pd.read_csv(csvpath, sep='\t', index_col=0) # type: ignore
    
    def label_hate(self, dataset):
        # ## Process data
        label_map = {
                'Neutral': False,
                'AffiliationDirectedAbuse': True,
                'Slur': True,
                'PersonDirectedAbuse': False,
                'IdentityDirectedAbuse': True,
                'CounterSpeech': False
            }

        dataset.data['hate'] = dataset.data.annotation_Primary.map(label_map.get)

    def extract_target_groups(self, dataset):
        dataset.data['target_groups'] = dataset.data.annotation_Target.map(lambda x: [self.groups_norm.get(x,x)] if isinstance(x, str) and self.groups_norm.get(x,x) != '' else [])

    def rename_text_column(self, dataset):
        # Rename text col
        dataset.data.rename(columns={'meta_text': 'text'}, inplace=True)


class HatexplainLoader(DataLoader):
    """ HateXplain dataset """

    def load(self, dataset):
        print(f"\tLoading {dataset.name}...")
        lines = []
        fpath = os.path.join(dataset.dirpath, dataset.load_paths[0])
        with open(fpath) as f:
            json_data = json.load(f)
            
        for entry_id, entry in json_data.items():
            lines.append({'text_id': entry['post_id'], 
                          'text': ' '.join(entry['post_tokens']),
                          'targets': list(set([target for targets in [item['target'] for item in entry['annotators']] for target in targets if target not in ['None', 'Other']])),
                          'hate': any([item['label'] == 'hatespeech' for item in entry['annotators']])
                         })
        dataset.data = pd.DataFrame(lines)

    def extract_target_groups(self, dataset):
        dataset.data['target_groups'] = dataset.data['targets'].map(
            lambda x: [self.groups_norm.get(t.lower(), t.lower()) for t in x if self.groups_norm.get(t.lower(), t.lower()) != ''])


class Elsherief2021Loader(DataLoader):
    """ Load and process ElSherief+2021 """

    def load(self, dataset):
        print("\tLoading elsherief2021...")
        ## Load those annotated with target (implicit hate)
        implicit_targeted = pd.read_csv(os.path.join(dataset.dirpath, dataset.load_paths[0]), sep='\t')
        ## Load stage 1 annotations, get instances not labeled hateful
        stg1 = pd.read_csv(os.path.join(dataset.dirpath, dataset.load_paths[1]), sep='\t')
        not_hate = stg1[stg1['class'] == 'not_hate']
        ## Concatenate the non-hate in and rename columns, etc (fill in with implicit)
        dataset.data = pd.concat([implicit_targeted, not_hate]).reset_index()

    def label_hate(self, dataset):
        dataset.data['class'].fillna('implicit_hate', inplace=True)
        dataset.data['hate'] = dataset.data['class'] == 'implicit_hate'
         
    def extract_target_groups(self, dataset):
        ## Annotate target type
        dataset.data['target_groups'] = dataset.data.target.map(lambda x: [self.groups_norm.get(x.lower(),x.lower())] if isinstance(x, str) and self.groups_norm.get(x.lower(),x.lower()) != '' else [])
        
    def rename_text_column(self, dataset):
        """ Rename text column """
        dataset.data.rename(columns={'post': 'text'}, inplace=True)


class SbicLoader(DataLoader):
    """ Social Bias Inference Corpus """
    
    def load(self, dataset):
        print("\tLoading sbic...")
        folds = []
        # Combine training, dev and test sets
        for fname in dataset.load_paths:
            folds.append(pd.read_csv(os.path.join(dataset.dirpath, fname), index_col=0))
            
        dataset.data = pd.concat(folds).reset_index()
            
    def label_hate(self, dataset):
        """ Label hate binary column """
        dataset.data['hate'] = dataset.data['offensiveYN'] > 0.5 # this follows the paper's threshold

    def extract_target_groups(self, dataset):
        dataset.data['target_groups'] = dataset.data['targetMinority'].map(self.flatten_targets)

    def rename_text_column(self, dataset):
        """ Rename text column """
        dataset.data.rename(columns={'post': 'text'}, inplace=True)

    def flatten_targets(self, target_str):
        """ Flatten target groups, returns list of all unique targets """
        flattened = set()
        for targets in eval(target_str):
            for target in targets.split(', '):
                normed = self.groups_norm.get(target.lower(), target.lower())
                if normed != '':
                    flattened.add(normed)
        # Combine 'men/women' -> people if both of these genders are present
        if 'lesbian women' in flattened and 'gay men' in flattened:
            flattened.remove('lesbian women')
            flattened.remove('gay men')
            flattened.add('gay people')
        for term in ['bisexual', 'transgender']:
            if f'{term} men' in flattened and f'{term} women' in flattened:
                for gender in ['men', 'women']:
                    flattened.remove(f'{term} {gender}')
                flattened.add(f'{term} people')
        return list(flattened)
        

class Salminen2018Loader(DataLoader):
    """ Salminen+2018 """

    def extract_target_groups(self, dataset):
        """ Extract target groups column """
        subcols = [col for col in dataset.data.columns if 'Sub' in col]
        dataset.data['subs'] = dataset.data[subcols].agg(lambda x: [el for el in x if isinstance(el, str)], axis=1)
        dataset.data['target_groups'] = dataset.data['subs'].map(lambda x: [
            self.groups_norm.get(self.extract_group(t), self.extract_group(t)) for t in x if self.groups_norm.get(self.extract_group(t), self.extract_group(t)) != ''])
        
    def label_hate(self, dataset):
        """ Label binary hate speech column """
        dataset.data['hate'] = dataset.data['Class']=='Hateful'
        
    @classmethod
    def extract_group(cls, target):
        return target.lower().replace('racist towards', '').replace('towards', '').strip()
        
    def rename_text_column(self, dataset):
        """ Rename text column """
        dataset.data.rename(columns={'message': 'text'}, inplace=True)  


class CivilcommentsLoader(DataLoader):
    """ CivilCommentsIdentities dataset """

    def extract_target_groups(self, dataset):
        """ Extract target groups column """
        target_cols = dataset.data.columns[dataset.data.columns.tolist().index('male'):-3]
        exclude_cols = ['other_religion', 'other_race_or_ethnicity']
        target_cols = [el for el in target_cols if el not in exclude_cols]

        threshold = 0.5 # minimum proportion of raters that labeled that target group for it to count
        dataset.data['targets'] = dataset.data[target_cols].values.tolist()
        dataset.data['targets'] = dataset.data['targets'].apply(lambda x: [target_cols[i] for i, el in zip(range(len(target_cols)), x) if el>=threshold])
        dataset.data['target_groups'] = dataset.data['targets'].map(lambda x: [self.groups_norm.get(t, t) for t in x])

    def label_hate(self, dataset):
        """ Label binary hate speech column """
        threshold = 0.5 # is maybe too low for the 'hate speech' label
        dataset.data['hate'] = dataset.data['toxicity']>=threshold 

    def rename_text_column(self, dataset):
        dataset.data.rename(columns={'comment_text': 'text'}, inplace=True)
