""" Logistic regression baseline for hate speech classification
@author Michael Miller Yoder
@year 2022
"""
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class LogisticRegressionClassifier:

    def train_eval(self, train, test):
        """ Train and evaluate on train and test dataframes """
        bow = {}

        # Build feature extractor
        vectorizer = TfidfVectorizer(min_df=1)
        bow['train'] = vectorizer.fit_transform(train['text'])
        bow['test'] = vectorizer.transform(test['text'])

        # Train, evaluate LR model 
        clf = LogisticRegression(solver='liblinear')
        clf.fit(bow['train'], train['hate'])
        preds = clf.predict(bow['test'])
        scores = pd.DataFrame(classification_report(test['hate'], preds, output_dict=True))

        return scores, preds
