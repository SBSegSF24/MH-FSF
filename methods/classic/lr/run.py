import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import logging
from utils import *

def add_arguments(parser):
    parser = parser.add_argument_group('Arguments for Linear Regression')
    parser.add_argument('--splits', type = int,
        help = 'Number of Splits. Default = 10',
        required = False, default = 10)

def run(args, path, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_lr
    logger_lr = logging.getLogger('LR')
    logger_lr.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_lr)
    print_message('Starting Features Selection', 'info', logger_lr)
    feature_names = np.array(X.columns.values.tolist())
    fold_count = [args.splits] * len(feature_names)
    fold_ft_num = list()
    fold_ft_to_delete = list()
    kf = KFold(n_splits = args.splits, shuffle = True, random_state = 0)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        coef_in = model.coef_
        n1 = len(coef_in)
        features_to_delete = list()

        for i in range(0, n1):
            if coef_in[i] < 0.1 and coef_in[i] > -0.1:
                features_to_delete.append(feature_names[i])
        fold_ft_num.append(len(features_to_delete))
        fold_ft_to_delete.append(features_to_delete)

        for ft in features_to_delete:
            index = list(feature_names).index(ft)
            fold_count[index] -= 1

    max_value = None
    index = None
    for idx, num in enumerate(fold_ft_num):
        if max_value is None or num > max_value:
            max_value = num
            index = idx

    reduced_dataset = X.drop(fold_ft_to_delete[index], axis = 1)
    reduced_dataset[args.class_column] = y
    output_file = os.path.join(args.output, f'lr_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_lr)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished Linear Regression Features Selection', 'info', logger_lr)
    return True
