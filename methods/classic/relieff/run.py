from ReliefF import ReliefF
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from termcolor import colored
import logging
from utils import *
import os

def add_arguments(parser):
    """
    Add arguments specific to ReliefF feature selection.

    Parameters:
    - parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for ReliefF')
    parser.add_argument( '-rfn', '--relieff-neighbors', type = int, default = 3,
        help = 'Number of Neighbors to Considered. Default: 3')

def run(args, path, dataset):
    """
    Run the ReliefF feature selection algorithm.

    Parameters:
    - args (argparse.Namespace): Arguments parsed from command line.
    - path (str): File path of the dataset.
    - dataset (pd.DataFrame): Input dataset.

    Returns:
    - bool: True if feature selection and saving the reduced dataset are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_relieff
    logger_relieff = logging.getLogger('ReliefF')
    logger_relieff.setLevel(logging.INFO)

    data, target = get_X_y(args, dataset, logger_relieff)
    X = data.values
    y = target.values
    n_features = int(data.shape[1] * args.threshold)
    print_message('Starting Features Selection', 'info', logger_relieff)
    fs = ReliefF(n_neighbors = args.relieff_neighbors, n_features_to_keep = n_features)
    fs.fit_transform(X, y)
    selected_features = list(data.columns[fs.top_features[:n_features]])
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'relieff_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_relieff)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished ReliefF Features Selection', 'info', logger_relieff)
    return True
