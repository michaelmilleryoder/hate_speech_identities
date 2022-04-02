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

import numpy as np

from data import Dataset


class Experiment:
    """ Class to run an 'experiment', just a run to get output """

    def __init__(self, datasets, calculate_control=True, classify=True, run_pca=True):
        """ Args:
                calculate_control [bool]: whether to calculate matching control terms to hegemonic terms
                classify [bool]: test classification with and without dominant groups vs control
                run_pca [bool]: run PCA on identity group splits
        """
        self.datasets = None
        self.calculate_control = calculate_control
        self.classify = classify
        self.run_pca = run_pca
        self.loaders = None

    def run(self):
        """ Run experiment """

        self.load_datasets()
        self.process_datasets()
        if calculate_control:
            self.get_control_terms()
        self.label_control()
        self.save_datasets()
        
    def load_datasets(self):
        """ Load datasets. Saves loaders to self.loaders """
        self.loaders = []
        for dataset in self.datasets:
            loader = dataset.loader() 
            loader.load(dataset)
            self.loaders.append(loader)

    def process_datasets(self):
        """ Process datasets """
        for dataset, loader in zip(self.datasets, self.loaders):
            loader.process(dataset)
    
    def label_control(self):
        """ Label control instances in datasets. This is separate from processing since
            calculating control terms depends on other processing. """
        for dataset, loader in zip(self.datasets, self.loaders):
            loader.label_control(dataset)

    def save_datasets(self):
        """ Save processsed datasets """
        for dataset, loader in zip(self.datasets, self.loaders):
            loader.save(dataset)

    def get_control_terms(self):
        """ Get, save out marginalized terms with similar frequencies across datasets for comparison
            to hegemonic terms
        """
        
        # Get frequncies of normalized labels across datasets
        group_targets = [] # target_group, group_label, dataset, count
        for dataset, loader in zip(self.datasets, self.loaders):
            target_counts = dataset.target_counts()
            target_counts['dataset'] = dataset.name
            target_counts['group_label'] = target_counts.group.map(lambda group: loader.group_labels.get(group, 'other'))
        target_dataset_counts = pd.concat(group_targets)
        target_dataset_counts.drop_duplicates(inplace=True)

        ## Get distributions of counts over datasets for normalized hegemonic labels
        heg_targets = target_dataset_counts.query('group_label == "hegemonic"')
        heg_counts = heg_targets.drop(columns=['group_label']).pivot_table(index=['group'], columns=['dataset'])
        log_heg_counts = heg_counts.apply(np.log2).replace(-np.inf, -1)
        log_heg_counts['magnitude'] = np.linalg.norm(log_heg_counts[[col for col in log_heg_counts.columns if col[0] == 'count']], axis=1)
        log_heg_counts = log_heg_counts.sort_values('magnitude', ascending=False).drop(columns='magnitude')
        
        # Find marginalized terms with similar frequency distributions across datasets as margemonic ones
        marg_targets = target_dataset_counts.query('group_label == "marginalized"')
        marg_counts = marg_targets.drop(columns=['group_label']).pivot_table(index=['group'], columns=['dataset'])
        log_marg_counts = marg_counts.apply(np.log2).replace(-np.inf, -1)
        marg = log_marg_counts.copy()
        control_terms = []
        for heg_term, heg_vec in log_heg_counts.iterrows():
            distances = np.linalg.norm(marg.values - heg_vec.values, axis=1)
            closest_marg = marg.index[np.argmin(distances)]
            control_terms.append(closest_marg)
            marg.drop(closest_marg, inplace=True) 

        # Save control terms out
        outpath = '/storage2/mamille3/hegemonic_hate/control_identity_terms.txt'
        with open(outpath, 'w') as f:
            for term in control_terms:
                f.write(f'{term}\n')
        
        # Check counts across datasets for heg and control
        print("Heg counts compared with control counts")
        print(heg_counts.sum())
        print(heg_counts.sum().sum())
        print()
        print(marg_counts.loc[control_terms].sum())
        print(marg_counts.loc[control_terms].sum().sum())
        

def main():
    """ Run experiments """

    # Settings (could load from a config)
    calculate_control = True 
    classify = True
    run_pca = False

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

    experiment = Experiment(datasets, calculate_control=calculate_control, classify=classify, run_pca=run_pca)
    experiment.run()


if __name__ == '__main__':
    main()
