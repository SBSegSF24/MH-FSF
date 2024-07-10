import pandas as pd
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np
import logging
from utils import *

def add_arguments(parser):
    """
    Add arguments specific to ANOVA feature selection.

    Parameters:
    - parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for ANOVA')

def run(args, path, dataset):
    """
    Run ANOVA feature selection algorithm based on provided arguments.

    Parameters:
    - args (argparse.Namespace): Arguments parsed from command line.
    - path (str): File path of the dataset.
    - dataset (pd.DataFrame): Input dataset.

    Returns:
    - bool: True if feature selection and saving the reduced dataset are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_anova
    logger_anova = logging.getLogger('ANOVA')
    logger_anova.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_anova)
    print_message('Starting Features Selection', 'info', logger_anova)
    constant_features = find_constant_features(X)
    if constant_features:
        print_message(f'{len(constant_features)} Constant(s) Feature(s)', 'warn', logger_anova)
        # Remove Constant Features
        X = X.drop(columns = constant_features)

    n_features = int(X.shape[1] * args.threshold)
    fvalue_best = SelectKBest(f_classif, k = n_features)
    fvalue_best.fit(X, y)
    feature_importances = np.array(list(zip(X.columns, fvalue_best.scores_)))
    feature_importances = pd.DataFrame(feature_importances, columns = ['feature', 'importance'])
    feature_importances = feature_importances.sort_values(by = 'importance', ascending = False)
    feature_importances = feature_importances.head(n_features)
    selected_columns = list(feature_importances['feature'].values)
    selected_columns.append(args.class_column)
    reduced_dataset = dataset[selected_columns]
    output_file = os.path.join(args.output, f'anova_{os.path.basename(path)}')
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished ANOVA Features Selection', 'info', logger_anova)
    return True
