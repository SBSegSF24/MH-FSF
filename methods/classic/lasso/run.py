import pandas as pd
import os
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import logging
from utils import *

def add_arguments(parser):
    """
    Add arguments specific to LASSO feature selection.

    Parameters:
    - parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for LASSO')

def run(args, path, dataset):
    """
    Run LASSO feature selection algorithm based on provided arguments.

    Parameters:
    - args (argparse.Namespace): Arguments parsed from command line.
    - path (str): File path of the dataset.
    - dataset (pd.DataFrame): Input dataset.

    Returns:
    - bool: True if feature selection and saving the reduced dataset are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_lasso
    logger_lasso = logging.getLogger('LASSO')
    logger_lasso.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_lasso)
    print_message('Starting Features Selection', 'info', logger_lasso)
    lasso_cv = LassoCV(cv = 5, max_iter = 10000).fit(X, y)
    best_alpha = lasso_cv.alpha_
    lasso = Lasso(alpha = best_alpha, max_iter = 10000)
    lasso.fit(X, y)
    selected_features = list(X.columns[lasso.coef_ != 0])
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'lasso_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_lasso)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished LASSO Features Selection', 'info', logger_lasso)
    return True
