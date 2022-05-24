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
from removal_comparison import RemovalComparison
from load_process_datasets import DatasetsLoader
from cross_dataset import CrossDatasetExperiment


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

    # Run removal comparison (with-heg/no-heg for example)
    if config['removal_comparison']['run']:
        removal_comparison = RemovalComparison(datasets, 
            config['removal_comparison']['removal_groups'],
            create_splits=config['removal_comparison']['create_splits'], 
            hate_ratio=config['hate_ratio'],
            cv_runs=config['removal_comparison']['cv_runs'],
        )
        removal_comparison.run(config['classifier']['name'], config['classifier']['settings'])

    # Run identity split PCA
    if config['cross_dataset']['run']:
        cross_dataset = CrossDatasetExperiment(datasets, 
            config['cross_dataset']['grouping'],
            config['classifier']['name'], 
            config['classifier']['settings'],
            create_datasets=config['cross_dataset']['create_identity_datasets'], 
            hate_ratio=config['hate_ratio'], 
            combine=config['cross_dataset']['combine_datasets'],)
        cross_dataset.run()


if __name__ == '__main__':
    main()
