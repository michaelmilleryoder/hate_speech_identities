import pdb

import numpy as np
import pickle
import pdb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from scipy.stats import chisquare, ttest_rel, wilcoxon
from tqdm import tqdm

from data import Dataset
from split_datasets import ComparisonSplits


class HegComparison:
    """ Compare splits with and without hegemonic hate.
        Test whether including hate toward hegemonic affects hate speech classification performance
            Split data into with-hegemonic and no-hegemonic, with-control and no-control
            Run logistic regression classifiers on these splits, save out results
    """

    def __init__(self, datasets, create_splits: bool = False, hate_ratio: float = 0.3):
        """ Args:
                create_splits: whether to recreate splits. If False, will just load them
        """
        self.datasets = datasets
        self.create_splits = create_splits
        self.hate_ratio = hate_ratio
        self.comparisons = None

    def run(self):
        """ Run experiment """

        # Create dataset splits for heg/no-heg comparison with control/no-control
        if self.create_splits:
            self.create_dataset_splits()
        else:
            self.load_dataset_splits()

        # Train and evaluate LR classifier on dataset splits (heg/no-heg vs control/no-control)
        print("Training and evaluating LR classifiers on splits...")
        self.train_eval_lr()

    def create_dataset_splits(self):
        """ Create dataset splits of heg/no-heg vs control/no-control """
        self.view_heg_vs_control()
        self.comparisons = ComparisonSplits(self.datasets, self.hate_ratio)
        self.comparisons.create_heg_control()

    def load_dataset_splits(self):
        """ Load already created dataset splits """
        print("Loading dataset splits...")
        self.comparisons = ComparisonSplits(self.datasets, self.hate_ratio)
        self.comparisons.load_heg_control()

    def view_heg_vs_control(self):
        """ Print number of instances marked heg vs marked control per dataset """
        for dataset in self.datasets:
            n_hegemonic = len(dataset.data.query("group_label == 'hegemonic'"))
            n_control = len(dataset.data.query("in_control == True"))
            print(dataset.name)
            print(f'# hegemonic: {n_hegemonic}')
            print(f'# control: {n_control}')
            print()

    def train_eval_lr(self):
        """ Train and evaluate logistic regression classifiers on splits """
        f1_scores = {}
        for splits in ['hegsplits', 'controlsplits']:
            
            scores = {}
            f1_scores[splits] = [] # List of dicts with keys: dataset, split, f1 (to create df)
            sigs = []

            for dataset_name in tqdm(self.comparisons.splits):
                tqdm.write(dataset_name)

                vectorizer = {}
                data = {}
                bow = {}
                scores[dataset_name] = {}
                for split, df in self.comparisons.splits[dataset_name][splits].items():
                    data[split] = df

                    # Check for NaNs
                    if data[split]['text'].isnull().values.any():
                        pdb.set_trace()

                    # Build feature extractor
                    vectorizer[split] = TfidfVectorizer(min_df=1)

                    # Train, evaluate LR model 
                    clf = make_pipeline(vectorizer[split], LogisticRegression(solver='liblinear'))

                    scores[dataset_name][split] = []
                    for _ in range(5):
                        # TODO: since there are duplicates, this should be splitting on unique indices
                        f1s = cross_validate(clf, data[split]['text'], data[split]['hate'], scoring=['f1'], cv=2)['test_f1'].tolist()
                        scores[dataset_name][split] += f1s
                        # confusion_matrices[dataset] = {}

                    f1_scores[splits].append({'dataset': dataset_name, 'split': split, 'f1': np.mean(scores[dataset_name][split])})

                # print()

                splitnames = ['with_special', 'no_special']
                # T-test or Wilcoxon for significance
                # sig = wilcoxon(scores[dataset][splitnames[0]], scores[dataset][splitnames[1]])
                sig = ttest_rel(scores[dataset_name][splitnames[0]], scores[dataset_name][splitnames[1]])
                sigs.append({'dataset': dataset_name, 
                             f'{splitnames[1]} > {splitnames[0]}': np.mean(scores[dataset_name][splitnames[1]]) > np.mean(scores[dataset_name][splitnames[0]]), 
                             'p < 0.05': sig.pvalue < 0.05, 'pvalue': sig.pvalue, 'statistic': sig.statistic,})

            print(splits)
            f1_df = pd.DataFrame(f1_scores[splits])
            print(f1_df)
            print()
            sigs_df = pd.DataFrame(sigs)
            print(sigs_df)

            # Save out CV scores
            with open(f'/storage2/mamille3/hegemonic_hate/tmp/{splits}_5x2cv_scores.pkl', 'wb') as f:
                pickle.dump(scores, f)
            f1_df.to_csv(f'/storage2/mamille3/hegemonic_hate/tmp/{splits}_5x2cv_f1.csv')
            sigs_df.to_csv(f'/storage2/mamille3/hegemonic_hate/tmp/{splits}_5x2cv_sigs.csv')

            print('************************')
