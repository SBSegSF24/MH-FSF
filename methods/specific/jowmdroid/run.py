import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from termcolor import colored, cprint
import argparse
import sys
import os
import logging
from utils import *

def add_arguments(parser):
    """
    Add arguments specific to the JOWMDroid feature selection script to the provided ArgumentParser object.

    Parameters:
        parser (ArgumentParser): ArgumentParser object to add arguments to.
    """
    parser.add_argument('--ignore-mi', help = 'Ignore Mutual Information Before Calculating Weights',
        action = 'store_true')
    parser.add_argument( '-wt', '--weights-threshold', type = float, default = 0.2,
        help = 'Select Only Features with Weight Greater Than This Value. Default: 0.2')

def select_features_with_mi(X, y):
    """
    Perform feature selection using Mutual Information (MI) regression.

    Parameters:
        X (DataFrame): Features dataset.
        y (Series): Target variable.

    Returns:
        DataFrame: Selected features dataset based on MI scores.
    """
    mi_model = mutual_info_regression(X, y, random_state = 0)
    mi_scores = pd.DataFrame({'score': mi_model}, index = list(X.columns))
    selected_features = list(mi_scores[mi_scores['score'] > 0.0].index)
    return X[selected_features]

def get_weights_from_classifiers(X, y, classifiers):
    """
    Retrieve feature importance weights from a list of classifiers.

    Parameters:
        X (DataFrame): Features dataset.
        y (Series): Target variable.
        classifiers (dict): Dictionary mapping classifier names to their configurations.

    Returns:
        list: List of lists containing feature weights from each classifier.
    """
    weights_list = list()
    for classifier in classifiers.values():
        constructor = classifier['constructor']
        constructor.fit(X, y)
        weights_attribute = classifier['importance_metric']
        weights = getattr(constructor, weights_attribute)
        # ensuring weights is not a list of list
        weights = weights if isinstance(weights[0], np.float64) else weights[0]
        weights_list.append(list(weights))
    return weights_list

def get_normalized_weights_average(weights_list):
    """
    Normalize and average feature weights obtained from different classifiers.

    Parameters:
        weights_list (list): List of lists containing feature weights from classifiers.

    Returns:
        ndarray: Array of normalized average feature weights.
    """
    normalized_weights_list = list()
    for weights in weights_list:
        max_value = max(weights)
        min_value = min(weights)
        normalized_weights = list()
        denominator = max_value - min_value
        denominator_is_zero = False if max_value != min_value else True
        for weight in weights:
            value = 0.5 if denominator_is_zero else (weight - min_value) / denominator
            normalized_weights.append(value)
        normalized_weights_list.append(normalized_weights)
    return np.average(normalized_weights_list, axis = 0)

def run_jowmdroid(X, y, weight_classifiers):
    """
    Run the JOWMDroid feature selection algorithm to calculate feature weights.

    Parameters:
        X (DataFrame): Features dataset.
        y (Series): Target variable.
        weight_classifiers (dict): Dictionary mapping classifier names to their configurations.

    Returns:
        ndarray: Array of initial normalized feature weights.
    """
    global logger_jowmdroid
    print_message('Calculating Weights', 'info', logger_jowmdroid)
    weights_list = get_weights_from_classifiers(X, y, weight_classifiers)
    initial_weights = get_normalized_weights_average(weights_list)
    return initial_weights

def run(args, path, dataset):
    """
    Main function to execute the JOWMDroid feature selection process and save the reduced dataset.

    Parameters:
        args (Namespace): Parsed arguments.
        path (str): Path to the dataset.
        dataset (DataFrame): Input dataset.

    Returns:
        bool: True if feature selection and dataset saving are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_jowmdroid
    logger_jowmdroid = logging.getLogger('JOWMDroid')
    logger_jowmdroid.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_jowmdroid)
    print_message('Starting Features Selection', 'info', logger_jowmdroid)
    print_message(f'Number of Initial Features: {X.shape[1]}', 'info', logger_jowmdroid)
    if not args.ignore_mi:
        print_message('Calculating Mutual Information', 'info', logger_jowmdroid)
        X = select_features_with_mi(X, y)
        print_message(f'Number of Features After MI: {X.shape[1]}', 'info', logger_jowmdroid)
    weight_classifiers = {
        'SVM': {'constructor': SVC(kernel = 'linear', random_state = 0), 'importance_metric': 'coef_'},
        'RF': {'constructor': RandomForestClassifier(random_state = 0), 'importance_metric': 'feature_importances_'},
        'LR': {'constructor': LogisticRegression(max_iter = 1000, random_state = 0), 'importance_metric': 'coef_'}}
    weights = run_jowmdroid(X, y, weight_classifiers)
    features = list(X.columns)
    feature_weights = pd.DataFrame({'weight': weights}, index = features)

    selected_features = list(feature_weights[feature_weights['weight'] >= args.weights_threshold].index)
    print_message(f'Number of Selected Features: {X.shape[1]}', 'info', logger_jowmdroid)
    selected_features.append(args.class_column)
    reduced_dataset = dataset[selected_features]
    output_file = os.path.join(args.output, f'jowmdroid_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_jowmdroid)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished JOWMDroid Features Selection', 'info', logger_jowmdroid)
    return True
