import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys
from random import choice
from argparse import ArgumentParser
from utils import *
import logging

def float_range(mini, maxi):
    """
    Helper function to create a float range checker for argument parsing.

    Parameters:
    mini (float): Minimum value.
    maxi (float): Maximum value.

    Returns:
    function: A function that checks if an argument is a float within the specified range.
    """
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError('Must be a Floating Point Number')
        if f <= mini or f >= maxi:
            raise argparse.ArgumentTypeError(f'Must be > {mini} and < {maxi}')
        return f
    return float_range_checker

def correlation_phase(X, y, k, method, methods, args):
    """
    Perform correlation analysis and drop highly correlated features.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.Series): Target variable.
    k (int): Number of features to keep.
    method (str): Name of the feature selection method.
    methods (dict): Dictionary of feature selection methods and their details.
    args (Namespace): Command-line arguments.

    Returns:
    pd.DataFrame: Reduced dataset after dropping correlated features.
    """
    global logger_sigapi
    feature_scores = methods[method]['function'](X, y, k)
    new_X = X[list(feature_scores['features'])]
    correlation = new_X.corr()
    model_RF=RandomForestClassifier()
    model_RF.fit(new_X,y)
    feats = dict()
    for feature, importance in zip(new_X.columns, model_RF.feature_importances_):
        feats[feature] = importance
    to_drop = set()

    for index in correlation.index:
        for column in correlation.columns:
            if index != column and correlation.loc[index, column] > 0.85:
               ft = column if feats[column] <= feats[index] else index
               to_drop.add(ft)

    print_message(f'Number of Features Removed: {len(to_drop)}', 'info', logger_sigapi)
    reduced_dataset = new_X.drop(columns = to_drop)
    reduced_dataset[args.class_column] = y
    return reduced_dataset

def add_arguments(parser):
    """
    Add command-line arguments for feature selection parameters.

    Parameters:
    parser (ArgumentParser): Argument parser object.
    """
    parser.add_argument('-diff', '--difference', type = float, default = 0.03,
        help = 'Difference Between Metrics. When All Metrics Are Less Than It, Selection Phase Finishes. Default: 0.03')
    parser.add_argument( '-ifp', '--initial-features-percent', type = float_range(0.0, 1.0), default = 0.1,
        help = 'Initial Features Percent. Default: 0.1')
    parser.add_argument('--sigapi-increment', type = float, default = 0.05,
        help = 'Percent to Increment Number of Features. Default: 0.05')

def get_moving_average(data, window_size = 5):
    """
    Compute the moving average of data.

    Parameters:
    data (np.array): Array of data.
    window_size (int): Size of the moving average window.

    Returns:
    np.array: Moving averages of the data.
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def get_minimal_range_suggestion(df, t = 0.001, window_size = 5):
    """
    Get a minimal range suggestion based on gradients of moving averages.

    Parameters:
    df (pd.DataFrame): Dataframe of values.
    t (float): Threshold for gradient differences.
    window_size (int): Size of the moving average window.

    Returns:
    int: Minimal range suggestion index.
    """
    moving_averages = np.array([get_moving_average(np.array(df)[:, i], window_size) for i in range(df.shape[1])]).T
    gradients = np.gradient(moving_averages, axis = 0)
    diffs = gradients[1:] - gradients[:-1]

    for i in range(len(diffs) - 1, 1, -1):
        if(any([diff > t for diff in diffs[i]])):
            return int(df.index[i])
    return -1

def calculateMutualInformationGain(features, target, k):
    """
    Calculate feature importance using Mutual Information Gain.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {'features': feature_names, 'score': mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by = ['score'], ascending = False)
    return df[:k]

def calculateRandomForestClassifier(features, target, k):
    """
    Select top k features using Random Forest Classifier feature importance.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = np.array(X.columns.values.tolist())
    test = RandomForestClassifier(random_state = 0)
    test = test.fit(X,y)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)), columns = ['features','score']).sort_values(by = ['score'], ascending = False)
    return df

def calculateExtraTreesClassifier(features, target, k):
    """
    Select top k features using Extra Trees Classifier feature importance.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = np.array(X.columns.values.tolist())
    test = ExtraTreesClassifier(random_state = 0)
    test = test.fit(X,y)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns = ['features','score']).sort_values(by = ['score'],ascending = False)
    return df

def calculateRFERandomForestClassifier(features, target, k):
    """
    Select top k features using RFE with Random Forest Classifier.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = np.array(X.columns.values.tolist())
    rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = k)
    model = rfe.fit(X,y)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ['features', 'score']).sort_values(by = ['score'], ascending = False)
    return df

def calculateRFEGradientBoostingClassifier(features, target, k):
    """
    Select top k features using RFE with Gradient Boosting Classifier.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names= np.array(X.columns.values.tolist())
    rfe = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = k)
    model = rfe.fit(X,y)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ['features', 'score']).sort_values(by = ['score'], ascending=False)
    return df

def calculateSelectKBest(features, target, k):
    """
    Select top k features using SelectKBest with chi-squared statistic.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names= np.array(features.columns.values.tolist())
    chi2_selector= SelectKBest(score_func = chi2, k= k)
    chi2_selector.fit(features,target)
    chi2_scores = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score'])
    df = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score']).sort_values(by = ['score'], ascending=False)
    return df[:k]

def calculateMetrics(new_X, y):
    new_X_train, new_X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3, random_state = 0)
    clf = RandomForestClassifier()
    clf.fit(new_X_train, y_train)
    prediction = clf.predict(new_X_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, zero_division = 0)
    recall = recall_score(y_test, prediction, zero_division = 0)
    f1 = f1_score(y_test, prediction, zero_division = 0)
    metrics = [accuracy, precision, recall, f1]
    return metrics

methods = { 'mutualInformation': { 'function': calculateMutualInformationGain, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectRandom': { 'function': calculateRandomForestClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectExtra': { 'function': calculateExtraTreesClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'RFERandom': { 'function': calculateRFERandomForestClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'RFEGradient': { 'function': calculateRFEGradientBoostingClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectKBest': { 'function': calculateSelectKBest, 'results': [[0,0,0,0,0]], 'is_stable': False }
}

def is_method_stable(previous_metrics, current_metrics, t = 0.03):
    """
    Check if a feature selection method is stable based on metric differences.

    Parameters:
    previous_metrics (np.array): Metrics of the previous iteration.
    current_metrics (np.array): Metrics of the current iteration.
    t (float): Threshold for metric differences.

    Returns:
    bool: True if method is stable, False otherwise.
    """
    differences = abs(current_metrics - previous_metrics)
    if(all(differences < t)):
        return True
    return False

def selection_phase(X, y, methods, args):
    """
    Perform feature selection using multiple methods until stability criteria are met.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.Series): Target variable.
    methods (dict): Dictionary of feature selection methods and their details.
    args (Namespace): Command-line arguments.

    Returns:
    str: Name of the most stable feature selection method.
    int: Number of features selected by the most stable method.
    """
    global logger_sigapi
    has_found_stable_method = False
    best_stable_method = None
    best_metric_value = 0
    total_features = len(X.columns)
    initial_n_features = int(args.initial_features_percent * total_features)
    features_increment = int(args.sigapi_increment * total_features)
    while initial_n_features < (total_features + features_increment) and not has_found_stable_method:
        k = total_features if initial_n_features > total_features else initial_n_features
        print_message(f'Number of Features: {k}', 'info', logger_sigapi)

        for method_name in methods.keys():
            feature_scores = methods[method_name]['function'](X, y, k)
            new_X = X[list(feature_scores['features'])]
            metrics =  calculateMetrics(new_X, y)
            methods[method_name]['results'] = np.append(methods[method_name]['results'], [[k,metrics[0], metrics[1], metrics[2], metrics[3]]], axis = 0)
            previous_metrics = methods[method_name]['results'][-2][1:]
            current_metrics = methods[method_name]['results'][-1][1:]

            if(len(methods[method_name]['results']) > 2 and is_method_stable(previous_metrics, current_metrics, args.difference)):
                has_found_stable_method = True
                accuracy = current_metrics[0]
                if(accuracy > best_metric_value):
                    best_metric_value = accuracy
                    best_stable_method = method_name
        initial_n_features += features_increment

    if(not has_found_stable_method):
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]['results'][-1][0])
    return best_stable_method, k

def run(args, path, dataset):
    """
    Main function to execute feature selection process and save reduced dataset.

    Parameters:
    args (Namespace): Command-line arguments.
    path (str): Path to dataset.
    dataset (str): Dataset name.

    Returns:
    bool: True if feature selection and dataset saving were successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_sigapi
    logger_sigapi = logging.getLogger('SigAPI')
    logger_sigapi.setLevel(logging.INFO)
    global X, y

    X, y = get_X_y(args, dataset, logger_sigapi)
    print_message('Starting Features Selection', 'info', logger_sigapi)
    best_stable_method, lower_bound = selection_phase(X, y, methods, args)
    print_message(f'Smallest Lower Limit Found: {best_stable_method}, {lower_bound}', 'info', logger_sigapi)
    print_message('Starting Correlation', 'info', logger_sigapi)
    reduced_dataset = correlation_phase(X, y, lower_bound, best_stable_method, methods, args)
    output_file = os.path.join(args.output, f'sigapi_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_sigapi)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished SigAPI Features Selection', 'info', logger_sigapi)
    return True
