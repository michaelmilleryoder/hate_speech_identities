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

def main():
    
    # Load datasets (if I modify this much, it should come from a config file or command line argument)
    datasets = [
        'kennedy2020',
    ]


if __name__ == '__main__':
    main()
