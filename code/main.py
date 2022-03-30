""" Main entry point to run experiments:
        1. Load, process datasets
        2. Test whether or not including hate toward hegemonic affects hate speech classification performance
            2.1 Split data into with-hegemonic and no-hegemonic as well as with-control and no-control
            2.2 Run logistic regression classifiers on these splits, save out results
        3. Visualize a PCA of what identity splits contain similar hate (prediction-wise)
            3.1 Create datasets of hate speech targeting specific identities
            3.2 Train and evaluate logistic regression classifiers within and across datasets
            3.3 Estimate and save out a PCA plot
"""

from load_data import DataLoader
from dataset import Dataset

def main():
    
    # Load datasets (if I modify this much, it should come from a config file or command line argument)
    datasets = [
        Kennedy2020Dataset('kennedy2020', fpaths=['ucberkeley-dlab/measuring-hate-speech', 'binary']),
        ElSherief2021Dataset('elsherief2021', fpaths=[
            '/storage2/mamille3/data/hate_speech/elsherief2021/implicit_hate_v1_stg3_posts.tsv',        
            '/storage2/mamille3/data/hate_speech/elsherief2021/implicit_hate_v1_stg1_posts.tsv',
        ], pandas_args='sep="\t"')
        Dataset('salminen2018', fpaths=['/storage2/mamille3/data/hate_speech/salminen2018/salminen2018.csv'], 
            pandas_args="index_col=0"),
        Dataset('sbic', fpaths=[
            '/storage2/mamille3/data/hate_speech/sbic/SBIC.v2.agg.trn.csv',
            '/storage2/mamille3/data/hate_speech/sbic/SBIC.v2.agg.dev.csv',
            '/storage2/mamille3/data/hate_speech/sbic/SBIC.v2.agg.tst.csv',
        ], pandas_args="index_col=0"),
        Dataset('cad', fpaths=['/storage2/mamille3/data/hate_speech/cad/cad_v1_1.tsv']),
    ]


    loader = DataLoader()

    for dataset in datasets:
        loader.load(dataset)


if __name__ == '__main__':
    main()
