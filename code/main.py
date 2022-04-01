""" Main entry point to run experiments:
        1. Load, process datasets
        2. Test whether ncluding hate toward hegemonic affects hate speech classification performance
            2.1 Split data into with-hegemonic and no-hegemonic, with-control and no-control
            2.2 Run logistic regression classifiers on these splits, save out results
        3. Visualize a PCA of what identity splits contain similar hate (prediction-wise)
            3.1 Create datasets of hate speech targeting specific identities
            3.2 Train and evaluate logistic regression classifiers within and across datasets
            3.3 Estimate and save out a PCA plot
"""

from data import Dataset


def main():
    """ Run experiments """

    # Load datasets (if I modify this much, it should come from a config file or command line argument)
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


    for dataset in datasets:
        loader = dataset.loader() 
        loader.load(dataset)
        loader.save(dataset)


if __name__ == '__main__':
    main()
