import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import logging
from utils import *

def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for Information Gain")

def run(args, path, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_ig
    logger_ig = logging.getLogger('IG')
    logger_ig.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_ig)
    print_message('Starting Features Selection', 'info', logger_ig)
    importances = mutual_info_classif(X, y)
    feature_importances = np.array(list(zip(X.columns, importances)))
    feature_importances = pd.DataFrame(feature_importances, columns = ['feature', 'importance'])
    feature_importances = feature_importances.sort_values(by = 'importance', ascending = False)

    n_features = int(X.shape[1] * args.threshold)
    feature_importances = feature_importances.head(n_features)
    selected_features = list(feature_importances['feature'].values)
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'ig_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_ig)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished IG Features Selection', 'info', logger_ig)
    return True
