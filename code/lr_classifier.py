""" Logistic regression baseline for hate speech classification
@author Michael Miller Yoder
@year 2022
"""
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class LogisticRegressionClassifier:
    """ A logistic regression classifier using sklearn """

    def __init__(self):
        self.clf = LogisticRegression(solver='liblinear')
        self.vectorizer = TfidfVectorizer(min_df=1)
        self.feats = {}

    def train(self, train):
        """ Define classifier, train on a training set 
            Args:
                train: DataFrame with text in a 'text' column and class value of boolean 'hate' column
        """
        self.feats['train'] = self.vectorizer.fit_transform(train['text'])
        # Train, evaluate LR model 
        self.clf.fit(self.feats['train'], train['hate'])

    def train_eval(self, train, test):
        """ Train and evaluate on train and test dataframes """
        self.train(train)
        scores, preds = self.eval(test)
        return scores, preds

    def eval(self, test):
        """ Evaluate on a test dataframe """
        self.feats['test'] = self.vectorizer.transform(test['text'])
        preds = self.clf.predict(self.feats['test'])
        scores = pd.DataFrame(classification_report(test['hate'], preds, output_dict=True))
        return scores, preds
