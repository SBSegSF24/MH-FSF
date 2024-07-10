import pandas as pd
import os
import numpy as np
import logging
from utils import *

def add_arguments(parser):
    parser = parser.add_argument_group('Arguments for Mean Absolute Deviation')

def run(args, path, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_mad
    logger_mad = logging.getLogger('MAD')
    logger_mad.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_mad)
    print_message('Starting Features Selection', 'info', logger_mad)
    mean_abs_diff = np.sum(np.abs(X - X.mean()), axis = 0) / X.shape[0]
    feature_mad = pd.DataFrame(mean_abs_diff, columns = ['mad']).reset_index()
    feature_mad.rename(columns = {'index': 'feature'}, inplace = True)
    feature_mad = feature_mad.sort_values(by = 'mad', ascending = False)
    n_features = int(X.shape[1] * args.threshold)
    feature_mad = feature_mad.head(n_features)
    selected_features = list(feature_mad['feature'].values)
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'mad_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_mad)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished MAD Features Selection', 'info', logger_mad)

    return True
