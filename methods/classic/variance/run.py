import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold
import logging
from utils import *

def add_arguments(parser):
    parser = parser.add_argument_group('Arguments for Variance Threshold')

def run(args, path, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    logger_variance = logging.getLogger('Variance')
    logger_variance.setLevel(logging.INFO)

    data, target = get_X_y(args, dataset, logger_variance)
    X = data.values
    print_message('Starting Features Selection', 'info', logger_variance)
    variances = X.var(axis = 0)
    sorted_indices = np.argsort(variances)[::-1]
    n_features = int(X.shape[1] * args.threshold)
    selected_features = list(data.columns[sorted_indices[:n_features]])
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'variance_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_variance)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished Variance Features Selection', 'info', logger_variance)
    return True
