import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
from utils import *

def add_arguments(parser):
    """
    Add arguments specific to PCA feature selection.

    Parameters:
    - parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for PCA')

def run(args, path, dataset):
    """
    Run PCA feature selection algorithm based on provided arguments.

    Parameters:
    - args (argparse.Namespace): Arguments parsed from command line.
    - path (str): File path of the dataset.
    - dataset (pd.DataFrame): Input dataset.

    Returns:
    - bool: True if feature selection and saving the reduced dataset are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_pca
    logger_pca = logging.getLogger('PCA')
    logger_pca.setLevel(logging.INFO)

    data, target = get_X_y(args, dataset, logger_pca)
    X = data.values
    y = target.values
    n_features = int(X.shape[1] * args.threshold)
    print_message('Starting Features Selection', 'info', logger_pca)
    pca = PCA(n_components = n_features)
    pca.fit(X)
    components = np.abs(pca.components_)
    # Sum Absolute Loads for Each Feature
    feature_importance = components.sum(axis = 0)
    important_features_indices = np.argsort(feature_importance)[::-1]
    # Select Principal Features
    selected_features = list(data.columns[important_features_indices[:n_features]])
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'pca_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_pca)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished PCA Features Selection', 'info', logger_pca)
    return True
