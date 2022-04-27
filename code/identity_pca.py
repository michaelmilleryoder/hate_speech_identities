""" Run a PCA on predictions from different identity target datasets """

import pickle
import pdb
import itertools
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import plotly.express as px

from create_identity_datasets import IdentityDatasetCreator
from bert_classifier import BertClassifier
from lr_classifier import LogisticRegressionClassifier


class IdentityPCA:

    def __init__(self, processed_datasets, clf_name, clf_settings, combine: bool = False, 
            create_datasets = False, 
            hate_ratio: float = 0.3, 
            incremental: bool = False):
        """ Args:
                clf_name: {bert, lr}
                clf_settings: dictionary of settings for the classifier
                create_datasets: whether to recreate both separate and combined datasets. 
                    If False, will just load them
                combine: whether to combine identity datasets across dataset sources
                incremental: whether to calculate PCA and save out results incrementally
        """
        self.processed_datasets = processed_datasets
        self.clf_name = clf_name
        self.clf_settings = clf_settings
        self.combine = combine
        self.create_datasets = create_datasets # False or list of either or both 'separate', 'combined'
        self.hate_ratio = hate_ratio
        self.incremental = incremental
        self.sep_identity_datasets = None # separate identity datasets
        self.expanded_datasets = None
        self.combined_identity_datasets = None
        self.scores = []
        self.group_labels = None
        self.reduced = None
        self.ic = None # IdentityDatasetCreator

    def run(self):
        """ Main function to run PCA """
        # Create or load identity-labeled datasets
        self.load_sep_identity_datasets()

        # Test viability of combining identity datasets
        viable, potential = self.test_combined()
        if self.combine:
            if len(viable) == 0:
                n_instances = [df.instance_count.sum() for df in list(potential.values())]
                selected_datasets = list(potential.keys())[n_instances.index(max(n_instances))]
            else:
                n_instances = [df.instance_count.sum() for df in list(viable.values())]
                selected_datasets = list(viable.keys())[n_instances.index(max(n_instances))]
            self.load_combined_identity_datasets(selected_datasets)

        # Run cross-dataset predictions and run PCA
        self.cross_dataset_eval()
        self.run_pca()

    def test_combined(self):
        """ Test if there are any sets of datasets that could be combined uniformly to have enough
            hate against hegemonic categories to plot a PCA combined by identity group """

        self.load_group_labels()

        dfs = [self.expanded_datasets[name].query('hate')[['target_groups', 'identity_group']] for name in sorted(self.expanded_datasets)]
        combined = pd.concat(dfs, keys=sorted(self.expanded_datasets.keys()), names=['dataset', 'text_id']).reset_index(level='dataset')
        combined['group_label'] = combined['identity_group'].map(self.group_labels.get)
        heg_counts = combined.query('group_label == "hegemonic"').groupby(['identity_group', 'dataset']).count().sort_values(['identity_group', 'group_label'], ascending=False).drop(columns='group_label').rename(columns={'target_groups': 'instance_count'})

        dataset_names = self.expanded_datasets.keys()
        n_datasets_range = range(3, 7)
        combos = []
        for i in n_datasets_range:
            combos.extend(list(itertools.combinations(dataset_names, i)))

        min_hegemonic_categories = 3
        min_combined_instances = 900
        max_oversample = 2 # maximum multiplier for oversampling small datasets
        possible_dataset_combos = set()
        combo_counts = {}
        potential = {}
        viable = {}

        for datasets in list(combos):
            selected = heg_counts.loc[heg_counts.index.get_level_values('dataset').isin(datasets)]
            counts = selected.groupby(selected.index.get_level_values('identity_group')).count()
            
            # Count how many hegemonic categories these datasets would cover
            avail_counts = counts[counts['instance_count']==len(datasets)]
            if len(avail_counts) >= min_hegemonic_categories:
                possible_dataset_combos.add(datasets)
                possible_combined = selected[selected.index.get_level_values('identity_group').isin(avail_counts.index)]

                # Calculate how many instances could be in combined datasets 
                combined_count = possible_combined.groupby(
                    possible_combined.index.get_level_values('identity_group')).agg(
                    {'instance_count': lambda x: min(x) * max_oversample * len(datasets)})
                combo_counts[datasets] = combined_count
                viable_combined = combined_count[combined_count['instance_count']>=min_combined_instances]
                if len(viable_combined) >= min_hegemonic_categories - 1:
                    potential[datasets] = combined_count.sort_values('instance_count', ascending=False).iloc[:min_hegemonic_categories]
                if len(viable_combined) >= min_hegemonic_categories:
                    viable[datasets] = combined_count.sort_values('instance_count', ascending=False).iloc[:min_hegemonic_categories]

        if len(viable) > 0:
            print(viable)
        else:
            print(f"No combinations of datasets give >{min_combined_instances} instances of hate for >={min_hegemonic_categories} identities (up to {max_oversample}x oversampled)")
            print(f"Closest is {potential}")

        return viable, potential
        
    def load_sep_identity_datasets(self):
        """ Load or create separate identity datasets """

        print("Creating/loading separate identity datasets...")
        self.ic = IdentityDatasetCreator(self.processed_datasets, self.hate_ratio, create=self.create_datasets)
        self.sep_identity_datasets, self.expanded_datasets = self.ic.create_sep_datasets()

    def load_combined_identity_datasets(self, selected_datasets):
        """ Load or create combined identity datasets 
            Args:
                selected_datasets: tuple of the names of datasets selected for the combinations
        """
        print("Creating/loading combined identity datasets...")
        self.combined_identity_datasets = self.ic.create_combined_datasets(selected_datasets)

    def cross_dataset_eval(self):
        """ Run cross-dataset predictions, save to self.scores """
        print("Cross-dataset training and evaluation...")
        scores = []

        if self.combine:
            datasets = self.combined_identity_datasets
        else:
            datasets = self.sep_identity_datasets
    
        for name, folds in tqdm(datasets.items(), ncols=100):
            tqdm.write(str(name))
            
            # Build classifier
            if self.clf_name == 'bert':
                clf = BertClassifier(**self.clf_settings)
            elif self.clf_name == 'lr':
                clf = LogisticRegressionClassifier()

            # Check for NaNs
            if folds['train']['text'].isnull().values.any():
                pdb.set_trace()
            if folds['test']['text'].isnull().values.any():
                pdb.set_trace()

            # Train model 
            clf.train(folds['train'])

            # Evaluate
            score_line = {'train_dataset': name} # a row for each test dataset
            
            for test_name, test_folds in datasets.items():
                test_scores, preds = clf.eval(test_folds['test'])
                score_line[test_name] = test_scores.loc['f1-score', 'True']
            scores.append(score_line)

            # Save out scores, run PCA incrementally
            if self.incremental and len(scores) > 2:
                self.scores = pd.DataFrame(scores).set_index('train_dataset')
                if self.combine:
                    scores_outpath = f'../output/combined_identity_{self.clf_name}_scores_{"+".join(self.ic.selected_datasets)}.csv'
                    self.scores.to_csv(scores_outpath)
                    tqdm.write(f"Saved cross-dataset scores to {scores_outpath}")
                self.run_pca()
                
        self.scores = pd.DataFrame(scores).set_index('train_dataset')
        scores_outpath = f'../output/combined_identity_{self.clf_name}_scores_{"+".join(self.ic.selected_datasets)}.csv'
        self.scores.to_csv(scores_outpath)
        tqdm.write(f"Saved cross-dataset scores to {scores_outpath}")

    def load_group_labels(self):
        """ Load group labels"""
        if self.group_labels is None:
            path = '../resources/group_labels.json'
            with open(path, 'r') as f:
                self.group_labels = json.load(f) 
    
    def run_pca(self):
        """ Run PCA over self.scores """

        self.load_group_labels()

        pca = PCA(n_components=2)
        self.reduced = pca.fit_transform(self.scores.values)
        self.reduced = pd.DataFrame(self.reduced, index=self.scores.index)

        # Assign group labels to groups so can visualize colors
        if self.combine:
            self.reduced['group_label'] = self.reduced.index.map(lambda x: self.group_labels.get(x))
        else:
            self.reduced['group_label'] = self.reduced.index.map(lambda x: self.group_labels.get(x[1]))

        # Plot
        if self.combine:
            title = f'Prediction weight PCA over combined identity datasets {self.ic.selected_datasets}'
        else:
            title = 'Prediction weight PCA over identities within datasets'
        fig = px.scatter(self.reduced, x=0, y=1, color='group_label', 
            text=self.reduced.index, width=1000, height=800,
            title=title)
        fig.update_traces(marker={'size': 20})
        fig.update_traces(textposition='top center')

        # Save out
        if self.combine:
            outname = f'combined_identity_{self.clf_name}_{"+".join(self.ic.selected_datasets)}_pca'
        else:
            outname = f'dataset_identity_{self.clf_name}_pca'
        outpath = f'../output/{outname}.png'
        fig.write_image(outpath)
        print(f"Saved dataset identity PCA to {outpath}")
