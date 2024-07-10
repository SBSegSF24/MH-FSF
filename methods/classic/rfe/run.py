from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
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
    Add arguments specific to RFE and RFECV feature selection.

    Parameters:
    - parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('-rfecv', '--rfe-cross-validation',
        help = 'Number of Features Selected is Tuned Automatically', action = 'store_true')
    parser.add_argument('-rfes', '--rfe-step', type = float, default = 0.01,
        help = 'Percent of Features to Remove at Each Iteration. Default: 0.01')

def run(args, path, dataset):
    """
    Run RFE or RFECV feature selection algorithm based on provided arguments.

    Parameters:
    - args (argparse.Namespace): Arguments parsed from command line.
    - path (str): File path of the dataset.
    - dataset (pd.DataFrame): Input dataset.

    Returns:
    - bool: True if feature selection and saving the reduced dataset are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_rfe
    logger_rfe = logging.getLogger('RFE')
    logger_rfe.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_rfe)
    step = int(X.shape[1] * args.rfe_step)
    step = step if step > 1 else 1
    print_message('Starting Features Selection', 'info', logger_rfe)
    if args.rfe_cross_validation:
        fs = RFECV(estimator = RandomForestClassifier(random_state = 0), step = step)
    else:
        fs = RFE(estimator = RandomForestClassifier(random_state = 0), n_features_to_select = args.threshold, step = step)

    fs.fit(X, y)
    optimal_features = fs.n_features_
    print_message(f'Number of Selected Features: {optimal_features}', 'info', logger_rfe)
    selected_columns = list(X.columns[fs.support_])
    selected_columns.append(args.class_column)
    reduced_dataset = dataset[selected_columns]
    output_file = os.path.join(args.output, f'rfe_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_rfe)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished RFE Features Selection', 'info', logger_rfe)
    return True
