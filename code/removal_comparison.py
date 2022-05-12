import pickle
import pdb

from sklearn.model_selection import cross_validate, GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from scipy.stats import chisquare, ttest_rel, wilcoxon
from tqdm import tqdm

from data import Dataset
from split_datasets import ComparisonSplits
from bert_classifier import BertClassifier
from lr_classifier import LogisticRegressionClassifier


class RemovalComparison:
    """ Compare splits with and without hegemonic hate.
        Test whether including hate toward hegemonic affects hate speech classification performance
            Split data into with-hegemonic and no-hegemonic, with-control and no-control
            Run logistic regression classifiers on these splits, save out results
    """

    def __init__(self, datasets, removal_groups, create_splits: bool = False, hate_ratio: float = 0.3, 
            cv_runs: int = 5):
        """ Args:
                create_splits: whether to recreate splits. If False, will just load them
                removal_groups: what sets of identities to remove (hegemonic, an identity category)
        """
        self.datasets = datasets
        self.removal_groups = removal_groups
        self.create_splits = create_splits
        self.hate_ratio = hate_ratio
        self.cv_runs = cv_runs
        self.comparisons = None

    def run(self, clf_name, clf_settings):
        """ Run experiment 
        Args:
            clf_name: One of {'bert', 'lr'}
            clf_settings: a dictionary of settings
        """

        # Create dataset splits for heg/no-heg comparison with control/no-control
        if self.create_splits:
            self.create_dataset_splits()
        else:
            self.load_dataset_splits()

        # Train and evaluate LR classifier on dataset splits (heg/no-heg vs control/no-control)
        print(f"Training and evaluating {clf_name} classifiers on splits...")
        self.train_eval_cv(clf_name, clf_settings)

    def create_dataset_splits(self):
        """ Create dataset splits of e.g. heg/no-heg vs control/no-control 
        """
        self.comparisons = ComparisonSplits(self.datasets, self.removal_groups, self.hate_ratio)
        self.comparisons.create_exp_control_splits()

    def load_dataset_splits(self):
        """ Load already created dataset splits """
        print("Loading dataset splits...")
        self.comparisons = ComparisonSplits(self.datasets, self.hate_ratio)
        self.comparisons.load_heg_control()

    def view_heg_vs_control(self):
        """ Print number of instances marked heg vs marked control per dataset """
        # TODO: should move to a function in split_datasets.py, combine with get_stats
        for dataset in self.datasets:
            n_hegemonic = len(dataset.data.query("group_label == 'hegemonic'"))
            n_control = len(dataset.data.query("in_control == True"))
            print(dataset.name)
            print(f'# hegemonic: {n_hegemonic}')
            print(f'# control: {n_control}')
            print()

    def train_eval_cv(self, clf_name, clf_settings):
        """ Train classifiers and evaluate with cross-validation """
        f1_scores = [] # List of dicts with keys: dataset, split, f1 (to create df)
        sigs = []
        scores = {}

        for dataset_name in tqdm(self.comparisons.splits, desc='datasets', ncols=100):
            tqdm.write(f'{dataset_name}')
            scores[dataset_name] = {}
            # cv_runs*2-fold CV * special/no-special (2) * expsplits/controlsplits (2)
            pbar = tqdm(total=self.cv_runs*2*2*2, desc='\tcv runs', ncols=80) 

            for splits in ['expsplits', 'controlsplits']:
                tqdm.write(f'\t{splits}')
                scores[dataset_name][splits] = {}
                split_f1 = {}

                for split, data in self.comparisons.splits[dataset_name][splits].items():
                    tqdm.write(f'\t\t{split}')
                    fold_scores_list = []

                    # Check for NaNs
                    assert not data['text'].isnull().values.any()

                    # Build classifier
                    if clf_name == 'bert':
                        clf = BertClassifier(**clf_settings)
                    elif clf_name == 'lr':
                        clf = LogisticRegressionClassifier()
                    #for _ in range(self.cv_runs):
                    # Define the fold splitter
                    splitter = GroupShuffleSplit(n_splits=self.cv_runs, test_size=0.5)
                    for inds0, inds1 in splitter.split(data, groups=data.index): # N times trying the CV
                        for setup in [(inds0, inds1), (inds1, inds0)]: # 2-fold CV
                            train = data.iloc[setup[0]]
                            test = data.iloc[setup[1]]

                            # Train and evaluate
                            fold_scores, preds = clf.train_eval(train, test)
                            pbar.update(1)
                            fold_scores_list.append(fold_scores)

                    concat = pd.concat(fold_scores_list) # could make this a more interpretable df
                    mean_scores = concat.groupby(concat.index).mean()
                    scores[dataset_name][splits][split] = concat
                    split_f1[split] = mean_scores.loc['f1-score', 'True']
                    f1_scores.append({'dataset': dataset_name, 'splits': splits, 'split': split, 
                        'f1': split_f1[split]})

                # T-test or Wilcoxon for significance
                # sig = wilcoxon(scores[dataset][splitnames[0]], scores[dataset][splitnames[1]])
                with_special_fold_scores = scores[dataset_name][splits]['with_special'].loc['f1-score', 'True']
                no_special_fold_scores = scores[dataset_name][splits]['no_special'].loc['f1-score', 'True']
                sig = ttest_rel(with_special_fold_scores, no_special_fold_scores) 
                sigs.append({'dataset': dataset_name, 'splits': splits,
                             'no_special > with_special': split_f1['no_special'] > split_f1['with_special'], 
                             'p < 0.05': sig.pvalue < 0.05, 'pvalue': sig.pvalue, 'statistic': sig.statistic,})

            # Save out CV scores (do after finishing every dataset)
            f1_df = pd.DataFrame(f1_scores)
            sigs_df = pd.DataFrame(sigs)
            #with open(f'../tmp/{clf_name}_{self.cv_runs}x2cv_scores.pkl', 'wb') as f:
            #    pickle.dump(scores, f)
            outstr = 'f../output/removal_comparison/{"_".join(self.removal_groups)}_{clf_name}_{self.cv_runs}x2cv_'
            f1_df.to_csv(outstr + '_f1.csv')
            sigs_df.to_csv(outstr + 'sig.csv')

            print(f1_df)
            print(sigs_df)
