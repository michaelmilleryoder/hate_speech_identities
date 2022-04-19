""" Main entry point to run experiments:
        1. Load, process datasets
        2. Test whether including hate toward hegemonic affects hate speech classification performance
            2.1 Split data into with-hegemonic and no-hegemonic, with-control and no-control
            2.2 Run logistic regression classifiers on these splits, save out results
        3. Visualize a PCA of what identity splits contain similar hate (prediction-wise)
            3.1 Create datasets of hate speech targeting specific identities
            3.2 Train and evaluate logistic regression classifiers within and across datasets
            3.3 Estimate and save out a PCA plot
"""

import yaml
import argparse
import pdb

import numpy as np
import pandas as pd

from data import Dataset
from split_datasets import ComparisonSplits
from heg_comparison import HegComparison
from load_process_datasets import DatasetsLoader
from identity_pca import IdentityPCA


def main():
    """ Run experiments """

    # Load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', nargs='?', type=str, help='file path to config file')
    args = parser.parse_args()
    with open(args.config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    # Load/process datasets
    datasets = [Dataset(name, load_paths=opts) for name, opts in config['datasets'].items()]
    if config['load_datasets']:
        print("Loading datasets...")
        loader = DatasetsLoader(datasets)
        loader.load_datasets(reprocess=config['reprocess_datasets'])

    # Run with-heg/no-heg comparison
    if config['heg_comparison']['run']:
        heg_comparison = HegComparison(datasets, 
            create_splits=config['heg_comparison']['create_splits'], hate_ratio=config['hate_ratio'])
        heg_comparison.run(config['clf_name'])

    # Run identity split PCA
    if config['pca']['run']:
        identity_pca = IdentityPCA(datasets, create_datasets=config['pca']['create_identity_datasets'], 
            hate_ratio=config['hate_ratio'], combine=config['pca']['combine_datasets'])
        identity_pca.run()


if __name__ == '__main__':
    main()
