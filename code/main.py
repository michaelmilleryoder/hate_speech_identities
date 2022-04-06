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

    # Settings (could load from a config)
    load_datasets = True # load raw or processed datasets instead of task-specific splits
    reprocess_datasets = False
    run_comparison = False
    hate_ratio = 0.3 # hate/non-hate ratio to sample each dataset
    create_splits = False # create comparison splits
    run_pca = True
    create_identity_datasets = False

    # Datasets (if I modify this much, it should come from a config file or command line argument)
    datasets = [
        Dataset('kennedy2020', 
            load_paths=['ucberkeley-dlab/measuring-hate-speech','binary']),
        Dataset('elsherief2021', 
            load_paths=[
            'implicit_hate_v1_stg3_posts.tsv',        
            'implicit_hate_v1_stg1_posts.tsv',
            ],
        ),
        Dataset('salminen2018'),
        Dataset('sbic',
             load_paths=[
            'SBIC.v2.agg.trn.csv',
            'SBIC.v2.agg.dev.csv',
            'SBIC.v2.agg.tst.csv',
        ]),
        Dataset('cad', 
            load_paths=['cad_v1_1.tsv']),
        Dataset('hatexplain',
            load_paths=['Data/dataset.json'],
        ),
    ]

    # Load/process datasets
    if load_datasets:
        print("Loading datasets...")
        loader = DatasetsLoader(datasets)
        loader.load_datasets(reprocess=reprocess_datasets)

    # Run with-heg/no-heg comparison
    if run_comparison:
        heg_comparison = HegComparison(datasets, create_splits=create_splits, hate_ratio=hate_ratio)
        heg_comparison.run()

    # Run identity split PCA
    if run_pca:
        identity_pca = IdentityPCA(datasets, create_datasets=create_identity_datasets, hate_ratio=hate_ratio)
        identity_pca.run()


if __name__ == '__main__':
    main()
