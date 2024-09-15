from sklearn import svm
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
import logging
from sklearn import metrics as skmetrics
from sklearn.metrics import confusion_matrix
import os
from termcolor import colored
from utils import *
from graphs import *

available_ml_models = {
    'svm': svm.SVC(),
    'rf': RandomForestClassifier(random_state = 0),
    'knn': KNeighborsClassifier()
}

def get_classifier(ml_model):
    """
    Returns the appropriate classifier based on the model name.

    Parameters:
    ml_model (str): Name of the machine learning model.

    Returns:
    clf (object): Instance of the classifier.
    """
    clf = available_ml_models.get(ml_model)
    return clf

def cross_validation(classifier, X, y, n_folds = 5):
    """
    Performs cross-validation and returns various performance metrics.

    Parameters:
    classifier (object): The classifier to be used.
    X (DataFrame): Feature set.
    y (Series): Label set.
    n_folds (int): Number of folds for cross-validation. Default is 5.

    Returns:
    metrics_results (dict): Dictionary containing various performance metrics.
    """
    y_pred = cross_val_predict(estimator = classifier, X = X, y = y, cv = n_folds)

    metrics_results = dict()
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics_results['tn'] = tn
    metrics_results['fp'] = fp
    metrics_results['fn'] = fn
    metrics_results['tp'] = tp
    metrics_results['accuracy'] = skmetrics.accuracy_score(y, y_pred)
    metrics_results['precision'] = skmetrics.precision_score(y, y_pred, zero_division = 0)
    metrics_results['recall'] = skmetrics.recall_score(y, y_pred, zero_division = 0)
    metrics_results['f1'] = skmetrics.f1_score(y, y_pred, zero_division = 0)
    metrics_results['roc_auc'] = skmetrics.roc_auc_score(y, y_pred)
    metrics_results['mcc'] = skmetrics.matthews_corrcoef(y, y_pred)
    return metrics_results

def plot_results(df, args, method, dataset):
    """
    Generates and saves various performance visualization graphs.

    Parameters:
    df (DataFrame): DataFrame containing the performance metrics.
    args (Namespace): Command-line arguments.
    method (str): Feature selection method.
    dataset (str): Name of the dataset file.
    """
    global logger
    print_message('Generating Metrics Graph', 'info', logger)
    graph_metrics(df, args, method, dataset)
    print_message('Generating Class Graph', 'info', logger)
    graph_class(df, args, method, dataset)
    print_message('Generating Radar Graph', 'info', logger)
    graph_radar(df, args, method, dataset)

def run_ml_models(args, models, method, dataset):
    """
    Runs machine learning models, evaluates performance, and generates visualization graphs.

    Parameters:
    args (Namespace): Command-line arguments.
    models (list): List of machine learning models to be evaluated.
    method (str): Feature selection method.
    dataset (str): Name of the dataset file.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger('Evaluation')
    logger.setLevel(logging.INFO)
    reduced_dataset_file = f'{method}_{os.path.basename(dataset)}'
    reduced_dataset_path = os.path.join(args.output, reduced_dataset_file)

    print_message(f'Loading Dataset {reduced_dataset_file}', 'info', logger)
    rds = get_dataset(args, reduced_dataset_path, logger)

    if args.class_column not in rds.columns:
        print_message(f'Expected Dataset {reduced_dataset_file} to Have a Class Column Named "{args.class_column}"', 'except', logger)
        exit(1)
    X, y = get_X_y(args, rds, logger)

    results = list()
    for model in models:
        clf = get_classifier(model)
        print_message(f'Running {model.upper()} to Dataset {reduced_dataset_file}', 'info', logger)
        results.append({**cross_validation(clf, X, y), 'model': model})
    results_file = os.path.join(args.output, f'evaluation_{reduced_dataset_file}')
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index = False)
    plot_results(results_df, args, method, dataset)
