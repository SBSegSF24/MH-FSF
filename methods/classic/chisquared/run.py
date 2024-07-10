import pandas as pd
import numpy as np
import os
import logging
from utils import *
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

def add_arguments(parser):
    parser = parser.add_argument_group('Arguments for Chi-Squared')

def chi2_features(X, y, args):
    th = int(X.shape[1] * args.threshold)
    chi2_selector = SelectKBest(chi2, k = th)
    chi2_selector.fit(X, y)
    chi2_scores = pd.DataFrame(list(zip(X.columns, chi2_selector.scores_)), columns = ['feature', 'score'])
    top_features = chi2_scores.sort_values(by = 'score', ascending = False)
    top_features = top_features.head(th)
    top_features = list(top_features['feature'].values)
    return top_features

def run(args, path, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_chi
    logger_chi = logging.getLogger('ChiSquared')
    logger_chi.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_chi)
    print_message('Starting Features Selection', 'info', logger_chi)
    selected_columns = chi2_features(X, y, args)
    selected_columns.append(args.class_column)
    reduced_dataset = dataset[selected_columns]
    output_file = os.path.join(args.output, f'chisquared_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_chi)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished ChiSquared Features Selection', 'info', logger_chi)
    return True
