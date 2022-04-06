""" Run a PCA on predictions from different identity target datasets """

import pickle
import pdb
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import plotly.express as px

from create_identity_datasets import IdentityDatasetCreator


class IdentityPCA:

    def __init__(self, processed_datasets, create_datasets: bool = False, hate_ratio: float = 0.3):
        """ Args:
                create_splits: whether to recreate splits. If False, will just load them
        """
        self.processed_datasets = processed_datasets
        self.create_datasets = create_datasets
        self.hate_ratio = hate_ratio
        self.identity_datasets = None
        self.scores = []
        self.group_labels = None
        self.reduced = None

    def run(self):
        """ Main function to run PCA """
        # Create identity datasets for heg/no-heg comparison with control/no-control
        if self.create_datasets:
            self.create_identity_datasets()
        else:
            self.load_identity_datasets()

        # Run LR cross-dataset predictions
        self.train_eval_lr()
    
        # Run, save out PCA
        self.run_pca()
        
    def load_identity_datasets(self):
        """ Load identity datasets that have already been saved out """
        print("Loading identity datasets...")
        path = f'/storage2/mamille3/hegemonic_hate/data/identity_splits_{self.hate_ratio}hate.pkl'
        with open(path, 'rb') as f:
            self.identity_datasets = pickle.load(f)

    def create_identity_datasets(self):
        """ Create identity datasets """
        print("Creating identity datasets...")
        ic = IdentityDatasetCreator(self.processed_datasets, self.hate_ratio)
        self.identity_datasets = ic.create_datasets()

    def train_eval_lr(self):
        """ Run logistic regression cross-dataset predictions, save to self.scores """
        clfs = {}
        scores = []
    
        for name, folds in tqdm(self.identity_datasets.items()):
            tqdm.write(str(name))
            
            # Extract features
            #tqdm.write('Extracting features...')
            bow = {}
            # Check for NaNs
            if folds['train']['text'].isnull().values.any():
                pdb.set_trace()
            if folds['test']['text'].isnull().values.any():
                pdb.set_trace()
            vectorizer = TfidfVectorizer(min_df=1)
            vectorizer.fit(folds['train']['text']) # corpus is a list of strings (documents)
            bow = {}
            bow['train'] = vectorizer.transform(folds['train']['text'])
            bow['test'] = vectorizer.transform(folds['test']['text'])
            bow.keys()

            # Train LR model 
            #tqdm.write('Training and evaluating model...')
            clfs[name] = LogisticRegression(solver='liblinear')
            clfs[name].fit(bow['train'], folds['train']['hate'])

            # Evaluate
            score_line = {'train_dataset': name}
            
            for test_name, test_folds in self.identity_datasets.items():
                test_bow = vectorizer.transform(test_folds['test']['text'])
                preds = clfs[name].predict(test_bow)
                score_line[test_name] = f1_score(test_folds['test']['hate'], preds)
            scores.append(score_line)
                
        self.scores = pd.DataFrame(scores).set_index('train_dataset')
    
    def run_pca(self):
        """ Run PCA over self.scores """

        # Load group labels
        path = '/storage2/mamille3/hegemonic_hate/resources/group_labels.json'
        with open(path, 'r') as f:
            self.group_labels = json.load(f) 

        pca = PCA(n_components=2)
        self.reduced = pca.fit_transform(self.scores.values)
        self.reduced = pd.DataFrame(self.reduced, index=self.scores.index)

        # Assign group labels to groups so can visualize colors
        self.reduced['group_label'] = self.reduced.index.map(lambda x: self.group_labels.get(x[1]))

        # Plot
        fig = px.scatter(self.reduced, x=0, y=1, color='group_label', 
            text=self.reduced.index, width=1000, height=800)
        fig.update_traces(marker={'size': 20})
        fig.update_traces(textposition='top center')

        # Save out
        outpath = '/storage2/mamille3/hegemonic_hate/output/dataset_identity_pca.png'
        fig.write_image(outpath)
        print(f"Saved dataset identity PCA to {outpath}")
