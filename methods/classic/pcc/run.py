import pandas as pd
import os
import argparse
import logging
from utils import *

def add_arguments(parser):
    parser = parser.add_argument_group('Arguments for Correlation Coefficient Selection')

def run(args, path, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_pcc
    logger_pcc = logging.getLogger('PCC')
    logger_pcc.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_pcc)
    print_message('Starting Features Selection', 'info', logger_pcc)
    correlations = X.apply(lambda x: x.corr(y))
    feature_correlations = pd.DataFrame(correlations, columns = ['correlation']).reset_index()
    feature_correlations.rename(columns = {'index': 'feature'}, inplace = True)
    feature_correlations['abs_correlation'] = feature_correlations['correlation'].abs()
    feature_correlations = feature_correlations.sort_values(by = 'abs_correlation', ascending = False)
    n_features = int(X.shape[1] * args.threshold)
    selected_features = feature_correlations.head(n_features)
    selected_features = list(selected_features['feature'].values)
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'pcc_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_pcc)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished PCC Features Selection', 'info', logger_pcc)
    return True
