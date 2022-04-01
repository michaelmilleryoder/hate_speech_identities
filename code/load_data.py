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

import numpy as np
import pandas as pd
import datasets # from HuggingFace

class DataLoader:
    """ Load, process datasets """

    def __init__(self):
        """ Constructor """
        self.groups_norm = None
        self.group_labels = None
        self.load_resources()

    def load_resources(self):
        """ Load resources for labeling """
        # Load group name normalization dict
        group_norm_path = '/storage2/mamille3/hegemonic_hate/resources/normalized_groups.json'
        with open(group_norm_path, 'r') as f:
            self.groups_norm = json.load(f) 

        # Load group labels dict
        group_label_path = '/storage2/mamille3/hegemonic_hate/resources/group_labels.json'
        with open(group_label_path, 'r') as f:
            self.group_labels = json.load(f) 

        # Make sure all normalized terms have group labels
        assert len([label for label in set(self.groups_norm.values()) if label not in self.group_labels]) == 0
        """
        # ## Control group terms

        path = '/storage2/mamille3/hegemonic_hate/control_identity_terms.txt'
        with open(path, 'r') as f:
            control_terms = f.read().splitlines()
        print(len(control_terms))
        control_terms
        """

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

    def load(self, dataset):
        """ Load a dataset to dataset.data. Usually overwritten by subclasses
        """
        print(f"Loading {dataset.name}")
        dataset.data = pd.read_csv(os.path.join(dataset.dirpath, dataset.load_paths[0]), index_col=0)

    def process(self, dataset):
        """ Do processing and naming of columns shared across datasets
         Each dataset dataframe will have a
         * index named 'text_id' with unique post/comment ID,
         * 'hate' binary hate_speech column, 
         * 'target_groups' list column of normalized identities
         * 'group_label' column {hegemonic, marginalized, other}
         * 'in_control' column: boolean whether a targeted group is in the control list of identity terms
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

        # Control group column
        #data['in_control'] = data.target_groups.map(lambda targets: any(t in control_terms for t in targets) if isinstance(targets, list) else False)

        # Drop nans in text column
        dataset.data = dataset.data.dropna(subset=['text'], how='any')

        # Check, rename index
        dataset.data.index.name = 'text_id'
        assert not dataset.data.index.duplicated(keep=False).any()

        print('\t\tdone.')

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

class Kennedy2020Loader(DataLoader):
    """ Load Kennedy+2020 dataset """

    def load(self, dataset):
        """ Load/download data """
        print("Loading kennedy2020...")
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
        dataset.data['target_groups'] = dataset.data[group_target_cols].progress_apply(self.extract_targets, axis=1)
        
    @classmethod
    def extract_group(colname: str):
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
            for colname in row[row==True].index]
        if len(targets) > 0:
            return targets
        else:
            return None
        
        
class CadLoader(DataLoader):
    """ Contextual Abuse Dataset (CAD) """

    def load(self, dataset):
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
        dataset.data['target_groups'] = dataset.data.annotation_Target.map(lambda x: [self.groups_norm.get(x,x)] if isinstance(x, str) else None)

    def rename_text_column(self, dataset):
        # Rename text col
        dataset.data.rename(columns={'meta_text': 'text'}, inplace=True)


class HatexplainLoader(DataLoader):
    """ HateXplain dataset """

    def load(self, dataset):
        print(f"Loading {dataset.name}...")
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
            lambda x: [self.groups_norm.get(t.lower(), t.lower()) for t in x])


class Elsherief2021Loader(DataLoader):
    """ Load and process ElSherief+2021 """

    def load(self, dataset):
        print("Loading elsherief2021...")
        ## Load those annotated with target (implicit hate)
        implicit_targeted = pd.read_csv(os.path.join(dataset.dirpath, dataset.load_paths[0]), sep='\t')
        ## Load stage 1 annotations, get instances not labeled hateful
        stg1 = pd.read_csv(os.path.join(dataset.dirpath, dataset.load_paths[1]), sep='\t')
        not_hate = stg1[stg1['class'] == 'not_hate']
        ## Concatenate the non-hate in and rename columns, etc (fill in with implicit)
        dataset.data = pd.concat([implicit_targeted, not_hate])

    def label_hate(self, dataset):
        dataset.data['class'].fillna('implicit_hate', inplace=True)
        dataset.data['hate'] = dataset.data['class'] == 'implicit_hate'
         
    def extract_target_groups(self, dataset):
        ## Annotate target type
        dataset.data['target_groups'] = dataset.data.target.map(lambda x: [self.groups_norm.get(x.lower(),x.lower())] if isinstance(x, str) else None)
        
        ## Control group column
        #data['in_control'] = data.target_groups.map(lambda targets: any(t in control_terms for t in targets) if isinstance(targets, list) else False)
        
    def rename_text_column(self, dataset):
        """ Rename text column """
        dataset.data.rename(columns={'post': 'text'}, inplace=True)


class SbicLoader(DataLoader):
    """ Social Bias Inference Corpus """
    
    def load(self, dataset):
        print("Loading sbic...")
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
        dataset.data.rename(columns={'post': 'text'}, inplace=True)

    def flatten_targets(self, target_str):
        """ Flatten target groups, returns list of all unique targets """
        flattened = set()
        for targets in eval(target_str):
            for target in targets.split(', '):
                flattened.add(self.groups_norm.get(target, target))
        return list(flattened)
        

class Salminen2018Loader(DataLoader):
    """ Salminen+2018 """

    def extract_target_groups(self, dataset):
        """ Extract target groups column """
        subcols = [col for col in dataset.data.columns if 'Sub' in col]
        dataset.data['subs'] = dataset.data[subcols].agg(lambda x: [el for el in x if isinstance(el, str)], axis=1)
        dataset.data['target_groups'] = dataset.data['subs'].map(lambda x: [
            self.groups_norm.get(self.extract_group(t), self.extract_group(t)) for t in x])
        
    def label_hate(self, dataset):
        """ Label binary hate speech column """
        dataset.data['hate'] = dataset.data['Class']=='Hateful'
        
    @classmethod
    def extract_group(target):
        return target.lower().replace('racist towards', '').replace('towards', '').strip()
        
        # Assign to control group
        #data['in_control'] = data.target_groups.map(lambda targets: any(t in control_terms for t in targets) if isinstance(targets, list) else False)
        
    def rename_text_column(self, dataset):
        """ Rename text column """
        pdb.set_trace()
        dataset.data.rename(columns={'message': 'text'}, inplace=True)  
        dataset.data.rename(columns={'post': 'text'}, inplace=True)
        

""" OLD JUPYTER NOTEBOOK CODE """
## ## Create data structures for use across all datasets
#
#
## Create data structure for holding value counts of group labels, targets across corpora
#group_label_distros = [] # dataset_name, total_items, targeted_items, hegemonic_count, marginalized_count, other_count
#group_target_distros = [] # target_group, group_label, dataset, count
#
## Data structure for capturing most frequent groups
#import pandas as pd
#
#top_groups = pd.DataFrame(columns=['corpus', 'split', 'top_groups'])
#top_groups
#
## Data structure for storing datasets (just targeted data)
## Each should have a
## * index named 'text_id' with unique post/comment ID,
## * 'hate' binary hate_speech column, 
## * 'target_groups' list column of normalized identities
## * 'group_label' column {hegemonic, marginalized, other}
## * 'in_control' column: boolean whether a targeted group is in the control list of identity terms
## * 'text' column
#hate_datasets = {}
#
#
## # HateXplain
#
## In[9]:
#
#
## Load data
#import json
#import pandas as pd
#
#lines = []
#
#fpath = '/storage2/mamille3/data/hate_speech/hatexplain/Data/dataset.json'
#with open(fpath) as f:
#    json_data = json.load(f)
#    
#for entry_id, entry in json_data.items():
#    lines.append({'text_id': entry['post_id'], 
#                  'text': ' '.join(entry['post_tokens']),
#                  'targets': list(set([target for targets in [item['target'] for item in entry['annotators']] for target in targets if target not in ['None', 'Other']])),
#                  'hate': any([item['label'] == 'hatespeech' for item in entry['annotators']])
#                 })
#data = pd.DataFrame(lines)
#data
#
#
## In[12]:
#
#
## Quickly find hegemonic percentage
#import numpy as np
#
#data['target_groups'] = data['targets'].map(lambda x: [groups_norm.get(t.lower(), t.lower()) for t in x])
#
#def assign_label(targets):
#    if isinstance(targets, float) or len(targets) == 0:
#        label = np.nan
#    else:
#        label = 'other'
#        labels = set([group_labels.get(target, 'other') for target in targets])
#        if 'marginalized' in labels and not 'hegemonic' in labels:
#            label = 'marginalized'
#        elif 'hegemonic' in labels:
#            label = 'hegemonic'
#        return label
#
#


## # Multilingual and Multi-aspect dataset (MLMA)
#
## In[35]:
#
#
#import os
#import pandas as pd
#
#dirpath = '/storage2/mamille3/data/hate_speech/mlma/'
#fpath = os.path.join(dirpath, 'en_dataset.csv')
#data = pd.read_csv(fpath, index_col=0)
#line = ['mlma_en', len(data)]
#data.head()
#
#
## In[36]:
#
#
#vc = data.group.value_counts()
#line.append(vc.sum())
#print(vc.sum())
#vc
#
#
## In[77]:
#
#
## View groups that are not already labeled
#vc[~vc.index.isin(group_labels)]
#
#
## In[37]:
#
#
#for label in ['hegemonic', 'marginalized', 'other']:
#    print(label)
#    n_instances_labeled = sum([count for group, count in vc.iteritems() if group_labels.get(group, 'other')==label])
#    print(f'{n_instances_labeled/sum(vc)} ({n_instances_labeled}/{sum(vc)})')
#    line.append(n_instances_labeled)
#
## Add counts to group_label_distros
#group_label_distros.append(line)
#print(line)
#
#
## In[38]:
#
#
## Get top groups in each split
#target_counts = vc
#n=5
#corpus = 'mlma_en'
#for label in ['hegemonic', 'marginalized', 'other']:
#    print(label)
#    selected_groups = [group for group in target_counts.index if group_labels.get(group)==label]
#    filtered = target_counts[selected_groups]
#    new_row = pd.DataFrame([[corpus, label, ', '.join(filtered.head(n).index.tolist())]], columns=top_groups.columns)
#    top_groups = pd.concat([top_groups, new_row], axis=0)
#    # print(filtered.head(n))
#    print()
#    
#top_groups
#
#
## In[39]:
#
#
#top_groups.reset_index(inplace=True)
#top_groups
#
#
## In[41]:
#
#
#for x in range(9,12):
#    top_groups.loc[x, 'corpus'] = 'mlma_en'
#top_groups 
#
#
## In[42]:
#
#
#top_groups.drop(columns='index', inplace=True)
#top_groups
#
#
## # ConvAbuse
## Just labels characteristic (like 'racist')
#
## In[36]:
#
#
## Load data
#import pandas as pd
#import os
#
#dirpath = '/storage2/mamille3/data/hate_speech/convabuse/'
#folds = []
#for fold in ['train', 'valid', 'test']:
#    folds.append(pd.read_csv(os.path.join(dirpath, '2_splits', f'ConvAbuseEMNLP{fold}.csv'), index_col=0))
#data = pd.concat(folds)
#data
#
#
## In[38]:
#
#
#data.columns.tolist()
#
#
## In[40]:
#
#
#data['Annotator2_target.generalised'].value_counts()
#
#
## # Plot distributions of group labels across datasets 
#
## In[24]:
#
#
#group_label_distros
#
#
## In[25]:
#
#
## Should have just made it a dataframe from the beginning
#distros = pd.DataFrame(group_label_distros, columns=['corpus', 'total_items', 'items_labeled_with_target', 'count_hegemonic', 'count_marginalized', 'count_other'])
#distros['count_hegemonic'] = distros['count_hegemonic']*-1 # for later graphs
#distros = distros.sort_values('count_hegemonic')
#distros
#
#
## In[177]:
#
#
## Convert to long-form df
#long_distros = pd.wide_to_long(distros, stubnames='count', sep='_',  i='corpus', j='split', suffix='\w+').reset_index()
#long_distros
#
#
## In[178]:
#
#
#selected = long_distros.query('split == "hegemonic" or split == "marginalized"').copy()
## selected.loc[selected['split']=='hegemonic', 'count'] = selected.loc[selected['split']=='hegemonic', 'count'] * -1
#selected
#
#
## In[179]:
#
#
## px.bar(selected, x='corpus', y='count', color='split', barmode='relative')
#px.bar(selected, y='corpus', x='count', color='split', barmode='relative', orientation='h')
#
#
## # Normalize target identities, sample control group of identity groups
#
## In[22]:
#
#
#target_dataset_counts = pd.DataFrame(group_target_distros)
#print(target_dataset_counts.dataset.unique())
#target_dataset_counts.drop_duplicates(inplace=True)
#target_dataset_counts
#
## View distributions of counts over datasets for normalized hegemonic labels
## Convert to wide format from long over dataset (pivot tables maybe?)
#heg_targets = target_dataset_counts.query('group_label == "hegemonic"')
#print(heg_targets.dataset.unique())
#
#heg_counts = heg_targets.drop(columns=['group_label']).pivot_table(index=['group'], columns=['dataset'])
#heg_counts
#
#heg_counts['group_normalized'] = heg_counts.index.map(lambda x: groups_norm.get(x, x))
#heg_counts
#
#heg_counts = heg_counts.groupby('group_normalized').agg({col: 'sum' for col in heg_counts.columns if col[0]=='count'})
#heg_counts
#
#log_heg_counts = heg_counts.apply(np.log2).replace(-np.inf, -1)
#log_heg_counts['magnitude'] = np.linalg.norm(log_heg_counts[[col for col in log_heg_counts.columns if col[0] == 'count']], axis=1)
#log_heg_counts = log_heg_counts.sort_values('magnitude', ascending=False).drop(columns='magnitude')
#log_heg_counts
#
#
## In[23]:
#
#
## Find marginalized terms with similar frequency distributions across datasets as margemonic ones
#marg_targets = target_dataset_counts.query('group_label == "marginalized"')
#print(marg_targets.dataset.unique())
#
#marg_counts = marg_targets.drop(columns=['group_label']).pivot_table(index=['group'], columns=['dataset'])
#marg_counts
#
#marg_counts['group_normalized'] = marg_counts.index.map(lambda x: groups_norm.get(x, x))
#marg_counts
#
#marg_counts = marg_counts.groupby('group_normalized').agg({col: 'sum' for col in marg_counts.columns if col[0]=='count'})
#marg_counts
#
#log_marg_counts = marg_counts.apply(np.log2).replace(-np.inf, -1)
#log_marg_counts
#
## Find closest match by Euclidean distance in marginalized terms
#from IPython.display import display
#
#marg = log_marg_counts.copy()
#control_terms = []
#
#for heg_term, heg_vec in log_heg_counts.iterrows():
#    distances = np.linalg.norm(marg.values - heg_vec.values, axis=1)
#    closest_marg = marg.index[np.argmin(distances)]
#    control_terms.append(closest_marg)
#    marg.drop(closest_marg, inplace=True) 
#    
#display(log_marg_counts.loc[control_terms])
#display(log_heg_counts)
#
## Save control terms out
#control_terms
#
#
## In[25]:
#
#
#outpath = '/storage2/mamille3/hegemonic_hate/control_identity_terms.txt'
#with open(outpath, 'w') as f:
#    for term in control_terms:
#        f.write(f'{term}\n')
#
#
## In[36]:
#
#
## Check counts across datasets for heg and control
#print(heg_counts.sum())
#print(heg_counts.sum().sum())
#print()
#print(marg_counts.loc[control_terms].sum())
#print(marg_counts.loc[control_terms].sum().sum())
#
#
## In[44]:
#
#
## Create boolean column of control/not
#for dataset in hate_datasets:
#    print(dataset)
#    hate_datasets[dataset]['in_control'] = hate_datasets[dataset]['target_groups'].map(lambda x: any([groups_norm.get(term, term) in control_terms for term in x]) if isinstance(x, list) else False)
#    print(hate_datasets[dataset].in_control.sum())
#
#
## In[129]:
#
#
## Check hegemonic instance counts (not sure why this isn't as close, but something to do with basing the matching off of term counts vs this is instances with any of the terms)
#for dataset in hate_datasets:
#    print(dataset)
#    print(sum(hate_datasets[dataset]['group_label']=='hegemonic'))
#
#
## In[78]:
#
#
## These might be more popular than the hegemonic ones, though, in which case it would create dataset splits that are more distinct
## Want them to be roughly as popular overall/for each dataset as the hegemonic set of labels is
#distros
#
#target_dataset_counts['group_normalized'] = target_dataset_counts.group.map(lambda x: groups_norm.get(x, x))
#target_dataset_counts['control_group'] = target_dataset_counts.group_normalized.isin(control_groups)
#control_group_instances = target_dataset_counts.query('control_group').groupby('dataset')['count'].sum()
#distros.join(control_group_instances)[['count_hegemonic', 'count']]
#
#
## In[79]:
#
#
## Look for marginalized identities that have similar frequency distributions to hegemonic terms
## Just get Euclidean distance between vectors of target groups (with maybe a log somewhere since kennedy2020 is so different?)
## Try individual matching between hegemonic terms and marginalized
#
#
## ## Sample control marginalized identities weighted by popularity
#
## In[60]:
#
#
## Normalize counts by dataset length
#distros = pd.DataFrame(group_label_distros, columns=['corpus', 'total_items', 'items_labeled_with_target', 'count_hegemonic', 'count_marginalized', 'count_other']).set_index('corpus')
#distros
#
#merged = target_dataset_counts.join(distros, on=['dataset']).drop(columns=['count_hegemonic', 'count_marginalized', 'count_other'])
#print(len(merged))
#print(len(target_dataset_counts))
#merged.head()
#
#merged['count_normalized'] = merged['count']/merged['items_labeled_with_target']
#merged.head()
#
## Group most popular identities first
#gped = merged.groupby('group').agg({'count_normalized': 'sum', 'group_label': 'first'}).sort_values('count_normalized', ascending=False)
#gped
#
#gped['group_normalized'] = gped.index.map(lambda x: groups_norm.get(x, x))
#
#def assign_label(labels):
#    label = 'other'
#    labels = set(labels)
#    if 'marginalized' in labels and not 'hegemonic' in labels:
#        label = 'marginalized'
#    elif 'hegemonic' in labels:
#        label = 'hegemonic'
#    return label
#
#gped_norm = gped.groupby('group_normalized').agg({'count_normalized': 'sum', 'group_label': assign_label}).sort_values('count_normalized', ascending=False)
#gped_norm
#
## Sample a group of identities for the control group
#possibilities = gped_norm.query('group_label == "marginalized"').iloc[:100]
#possibilities
#
#
## In[77]:
#
#
#control_groups = possibilities.sample(2, weights=possibilities.count_normalized, random_state=9).index
#control_groups
#
#
## In[78]:
#
#
## These might be more popular than the hegemonic ones, though, in which case it would create dataset splits that are more distinct
## Want them to be roughly as popular overall/for each dataset as the hegemonic set of labels is
#distros
#
#target_dataset_counts['group_normalized'] = target_dataset_counts.group.map(lambda x: groups_norm.get(x, x))
#target_dataset_counts['control_group'] = target_dataset_counts.group_normalized.isin(control_groups)
#control_group_instances = target_dataset_counts.query('control_group').groupby('dataset')['count'].sum()
#distros.join(control_group_instances)[['count_hegemonic', 'count']]
#
#
## # Group target identities
#
## In[22]:
#
#
#target_dataset_counts = pd.DataFrame(group_target_distros)
#print(target_dataset_counts.dataset.unique())
#target_dataset_counts.drop_duplicates(inplace=True)
#target_dataset_counts['group_normalized'] = target_dataset_counts.group.map(lambda x: groups_norm.get(x, x))
#selected = target_dataset_counts.query('count > 20').sort_values(['dataset', 'count'], ascending=False)
#selected
#
#
## In[3]:
#
#
#identity_groups = {
#    'white people': ['white people'],
#    'corporations': ['corporations'],
#    'minorities': ['people of color'],
#    'transgender men': ['lgbtq+ people', 'men'],
#    'people with physical disabilities': ['people with disabilities'],
#    'rape victims': ['victims of violence'],
#    'bisexual women': ['lgbtq+ people', 'women'],
#    'poor folks': ['working class people'],
#    'bisexual': ['lgbtq+ people'],
#    'native_american': ['indigenous people'],
#    'non_binary': ['lgbtq+ people'],
#    'non-binary people': ['lgbtq+ people'],
#    'mormon': ['mormons'],
#    'atheists': ['atheists'],
#    'teenagers': ['young people'],
#    'seniors': ['older people'],
#    'disability_hearing_impaired': ['people with disabilities'],
#    'disability_visually_impaired': ['people with disabilities'],
#    'young_adults': ['young people'],
#    'ethiopian people': ['black people'],
#    'bisexual men': ['lgbtq+ people', 'men'],
#    'sexual assault victims': ['victims of violence'],
#    'harassment victims': ['victims of violence'],
#    'children': ['young people'],
#    'old folks': ['older people'],
#    'orphans': ['young people'], # debatable,
#    'child rape victims': ['victims of violence'],
#    'child sexual assault victims': ['victims of violence'],
#    'genocide victims': ['victims of violence', 'people of color'],
#    'pedophilia victims': ['victims of violence'],
#    'kids': ['young people'],
#    'japanese': ['asian people'],
#    'japanese people': ['asian people'],
#    'holocaust survivors': ['victims of violence', 'jews'],
#    'child molestation victims': ['victims of violence'],
#    'priests': ['christians'],
#    'assault victims': ['victims of violence'],
#    'mass shooting victims': ['victims of violence'],
#    'terrorism victims': ['victims of violence'],
#    'lesbian women': ['lgbtq+ people', 'women'],
#    'holocaust victims': ['victims of violence', 'jews'],
#    'native american people': ['indigenous people', 'people of color'],
#    'black people': ['black people', 'people of color'],
#    'liberals': ['left-wing people'],
#    'progressives': ['left-wing people'],
#    'leftists': ['left-wing people'],
#    'mexican people': ['latinx people', 'people of color'],
#    'white men': ['white people', 'men'],
#    'conservative men': ['right-wing people', 'men'],
#    'white conservatives': ['right-wing people', 'white people'],
#    'antifa': ['left-wing extremists', 'left-wing people'],
#    'white liberals': ['white people', 'left-wing people'],
#    'germans': ['europeans'],
#    'arabic/middle eastern people': ['muslims and arabic/middle eastern people'],
#    'african people': ['black people'],
#    'middle eastern folks': ['arabic/middle eastern people'],
#    'refugees': ['immigrants'],
#    'people with mental disabilities': ['people with disabilities'],
#    'gay men': ['lgbtq+ people'],
#    'transgender people': ['lgbtq+ people'],
#    'muslims': ['muslims and arabic/middle eastern people'],
#    'involuntary celibates': ['right-wing extremists', 'right-wing people'], # debatable
#    'gay people': ['lgbtq+ people'],
#    'left-wing people (social justice)': ['left-wing people'],
#    'non-gender dysphoric transgender people': ['lgbtq+ people'],
#    'feminists': ['women'],
#    'chinese women': ['asian people', 'people of color', 'women'],
#    'democrats': ['left-wing people'],
#    'people with autism': ['people with disabilities'],
#    'chinese people': ['asian people', 'people of color'],
#    'police officers': ['military and law enforcement'],
#    'undocumented immigrants': ['immigrants'],
#    'activists (anti-fascist)': ['left-wing extremists'],
#    'donald trump supporters': ['right-wing people'],
#    'people from pakistan': ['asian people', 'people of color'],
#    'americans': ['americans'],
#    'elderly people': ['older people'],
#    'working class people': ['working class people'],
#    'people of color': ['people of color'],
#    'republicans': ['right-wing people'],
#    'convservatives': ['right-wing people'],
#    'people from mexico': ['latinx people', 'people of color'],
#    'gamers': ['gamers'],
#    'men': ['men'],
#    'indian people': ['asian people', 'people of color'],
#    'people with aspergers': ['people with disabilities'],
#    'activists (animal rights)': ['left-wing extremists', 'left-wing people'], # debatable
#    'rich people': ['rich people'],
#    'fans of anthropomorphic animals ("furries")': ['furries'],
#    'catholics': ['christians'],
#    'romani people': ['romani people'],
#    'transgender women': ['lgbtq+ people', 'women'],
#    'conservatives': ['right-wing people'],
#    'pregnant folks': ['women'], # controversial but I think this is what's implied
#    'bisexual people': ['lgbtq+ people'],
#    'hindus': ['hindus'],
#    'buddhist': ['buddhists'],
#    'straight people': ['straight people'],
#    'young adults': ['young people'],
#    'middle-aged people': ['middle-aged people'],
#    'communists': ['left-wing people'],
#}
#
#
## In[4]:
#
#
## Save out identity group assignments
#import json
#
#outpath = '/storage2/mamille3/hegemonic_hate/identity_groups.json'
#
#with open(outpath, 'w') as f:
#    json.dump(identity_groups, f)
#
#
## In[24]:
#
#
## Load identity group assignments
#import json
#
#path = '/storage2/mamille3/hegemonic_hate/identity_groups.json'
#
#with open(path, 'r') as f:
#    identity_groups = json.load(f)
#
#
## In[26]:
#
#
## Look at terms that I haven't grouped
#grouped_identities = set(identity_groups.keys()).union(set([val for vals in identity_groups.values() for val in vals]))
#grouped_identities
#
#
## In[31]:
#
#
#selected.loc[~selected.group_normalized.isin(grouped_identities)]
#
#
## In[27]:
#
#
## Counts of hate targeted at identity groups
#target_dataset_counts['identity_group'] = target_dataset_counts.group_normalized.map(lambda x: identity_groups.get(x, [x] if x in grouped_identities else []))
## .query('count > 20').sort_values(['dataset', 'count'], ascending=False)
#target_dataset_counts
#
#s = target_dataset_counts.identity_group.apply(pd.Series, 1).stack()
#s.index = s.index.droplevel(-1)
#s.name = 'identity_group'
#del target_dataset_counts['identity_group']
#target_group_counts = target_dataset_counts.join(s)
#target_group_counts
#
#
## In[29]:
#
#
## But really want 1000 instances of hate at that target (restricts kennedy+2020 and sbic some)
## Should just start from the hate_datasets
#
#threshold = 1000
#dataset_group_counts = target_group_counts.groupby(['dataset', 'identity_group'])['count'].sum()
#filtered = dataset_group_counts[dataset_group_counts >= threshold]
#print(len(filtered))
#print(len(filtered.index.get_level_values('identity_group').unique()))
#filtered
#
#
## In[1]:
#
#
## Save out (shouldn't be manual like this)
#selected_dataset_groups = [
#    ('elsherief2021', 'people of color'),
#    ('elsherief2021', 'white people'),
#    ('kennedy2020', 'asian people'),
#    ('kennedy2020', 'black people'),
#    ('kennedy2020', 'christians'),
#    ('kennedy2020', 'immigrants'),
#    ('kennedy2020', 'indigenous people'),
#    ('kennedy2020', 'jews'),
#    ('kennedy2020', 'latinx people'),
#    ('kennedy2020', 'lgbtq+ people'),
#    ('kennedy2020', 'men'),
#    ('kennedy2020', 'muslims and arabic/middle eastern people'),
#    ('kennedy2020', 'people of color'),
#    ('kennedy2020', 'people with disabilities'),
#    ('kennedy2020', 'straight people'),
#    ('kennedy2020', 'white people'),
#    ('kennedy2020', 'women'),
#    ('kennedy2020', 'young people'),
#    ('sbic', 'black people'),
#    ('sbic', 'jews'),
#    ('sbic', 'lgbtq+ people'),
#    ('sbic', 'people of color'),
#    ('sbic', 'people with disabilities'),
#    ('sbic', 'victims of violence'),
#    ('sbic', 'women'),
#]
#len(selected_dataset_groups)
#
#
## In[2]:
#
#
#import os
#import pickle
#
#outpath = '/storage2/mamille3/hegemonic_hate/tmp/selected_dataset_groups.pkl'
#with open(outpath, 'wb') as f:
#    pickle.dump(selected_dataset_groups, f)
