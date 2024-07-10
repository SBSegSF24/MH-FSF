from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from argparse import ArgumentParser
import sys
import os
import logging
from utils import *

heuristic_metrics = ['precision', 'accuracy', 'recall', 'f-measure']

def add_arguments(parser):
    """
    Add arguments specific to the RFG script to the provided ArgumentParser object.

    Parameters:
        parser (ArgumentParser): ArgumentParser object to add arguments to.
    """
    parser = parser.add_argument_group('Arguments for RFG')
    parser.add_argument("--n-list", type = int, nargs = '*',
        help="List With Number of Features to Test")
    parser.add_argument('--rfg-increment', type = float, default = 0.05,
        help = 'Percent of Increment Features. Default: 0.05')
    parser.add_argument('--features-only', action = 'store_true',
        help = 'Ignore Mutual Information Before Calculating Weights')
    parser.add_argument('-ht', '--heuristic-threshold', type = float, default = 0.95,
        help = 'Heuristic Metric Threshold for Selecting Best Dataset. Default: 0.95')
    parser.add_argument('-ds', '--decrement-step', type = float, default = 0.05,
        help = 'Decrement Applied to Heuristic Metric Threshold, If There is no Dataset. Default: 0.05')
    parser.add_argument('-hm', '--heuristic-metric', choices = heuristic_metrics, default = 'recall',
        help = f'Metric to Base Choice of Best Dataset of Selected Features. Options: {str(heuristic_metrics)}. Default: recall')

def run_experiment(X, y, args, classifiers, score_functions = [chi2, f_classif], n_list = list()):
    """
    Perform feature selection experiments and evaluate classifiers on selected features.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.
        classifiers (dict): Dictionary of classifiers to evaluate.
        score_functions (list): List of score functions for feature selection.
        n_list (list): List of number of features to test.

    Returns:
        DataFrame: Results dataframe containing evaluation metrics for each experiment.
        dict: Dictionary containing feature rankings for each score function.
    """
    global logger_rfg
    results = list()
    feature_rankings = dict()
    if n_list:
        n_values = n_list
    else:
        n_increment = int(X.shape[1] * args.rfg_increment)
        n_values = range(n_increment, X.shape[1], n_increment)
    for n in n_values:
        if(n > X.shape[1]):
            print_message(f'Skipping N = {n}, Since it\'s Greater Than Number of Features Available ({X.shape[1]})', 'warn', logger_rfg)
            continue

        print_message(f'Testing N = {n}', 'info', logger_rfg)
        for score_function in score_functions:
            if n == max(x for x in n_values if x <= X.shape[1]):
                selector = SelectKBest(score_func=score_function, k = n).fit(X, y)
                X_selected = X.iloc[:, selector.get_support(indices = True)].copy()
                feature_scores_sorted = pd.DataFrame(list(zip(X_selected.columns.values.tolist(), selector.scores_)), columns = ['features','score']).sort_values(by = ['score'], ascending = False)
                X_selected_sorted = X_selected.loc[:, list(feature_scores_sorted['features'])]
                X_selected_sorted['class'] = y
                feature_rankings[score_function.__name__] = X_selected_sorted
                if X_selected.shape[1] == 1:
                    print_message('No Features Selected', 'warn', logger_rfg)

            if args.features_only:
                continue
            X_selected = SelectKBest(score_func=score_function, k = n).fit_transform(X, y)
            kf = KFold(n_splits = 5, random_state = 0, shuffle = True)
            fold = 0
            for train_index, test_index in kf.split(X_selected):
                X_train, X_test = X_selected[train_index], X_selected[test_index]
                y_train, y_test = y[train_index], y[test_index]
                for classifier_name, classifier in classifiers.items():
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict = True, zero_division = 0)
                    results.append({'n_fold': fold,
                                    'n_features': n,
                                    'score_function': score_function.__name__,
                                    'algorithm': classifier_name,
                                    'accuracy': report['accuracy'],
                                    'precision': report['macro avg']['precision'],
                                    'recall': report['macro avg']['recall'],
                                    'f-measure': report['macro avg']['f1-score']
                                    })
                fold += 1
    return pd.DataFrame(results), feature_rankings

def get_best_result(results, args):
    """
    Identify the best feature set based on heuristic metrics and threshold.

    Parameters:
        results (DataFrame): Results dataframe from feature selection experiments.
        args (Namespace): Parsed arguments.

    Returns:
        tuple: Best number of features and score function based on heuristic criteria.
    """
    averages = results.groupby(['n_features','score_function']).mean(numeric_only = True).drop(columns = ['n_fold'])
    maximun_score = max(averages.max())
    th = args.heuristic_threshold
    while th > 0:
        for n, score_function in averages.index:
            if(averages.loc[(n, score_function)][args.heuristic_metric] > th * maximun_score):
                return n, score_function
        th -= args.decrement_step
    print_message(f'Unable to Find a Dataset. Try Again Varying Parameters', 'except', logger_rfg)

def get_best_features_dataset(best_result, feature_rankings, class_column):
    """
    Retrieve the best features dataset based on the best result from feature selection.

    Parameters:
        best_result (tuple): Tuple containing the best number of features and score function.
        feature_rankings (dict): Dictionary containing feature rankings for each score function.
        class_column (str): Name of the class column.

    Returns:
        DataFrame: Reduced dataset containing the best selected features and class column.
    """
    n_features, score_function = best_result
    X = feature_rankings[score_function].drop(columns = [class_column])
    y = feature_rankings[score_function][class_column]
    X_selected = X.iloc[:, :n_features]
    reduced_dataset = X_selected.join(y)
    return reduced_dataset

def run(args, path, dataset):
    """
    Main function to execute feature selection and dataset saving process.

    Parameters:
        args (Namespace): Parsed arguments.
        path (str): Path to the dataset.
        dataset (DataFrame): Input dataset.

    Returns:
        bool: True if feature selection and dataset saving are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_rfg
    logger_rfg = logging.getLogger('RFG')
    logger_rfg.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_rfg)
    print_message('Starting Features Selection', 'info', logger_rfg)
    constant_features = find_constant_features(X)
    if constant_features:
        print_message(f'{len(constant_features)} Constant(s) Feature(s)', 'warn', logger_rfg)
        #Remove Constant Features
        X.drop(columns = constant_features, inplace = True)
    n_list = args.n_list if args.n_list else list()

    classifiers = {
        'RandomForest': RandomForestClassifier(random_state = 0),
        'KNN': KNeighborsClassifier()
        }

    print_message(f'Running RFG', 'info', logger_rfg)
    results, feature_rankings = run_experiment(X, y, args, classifiers, n_list = n_list)

    print_message(f'Selecting Best Features', 'info', logger_rfg)
    best_result = get_best_result(results, args)
    reduced_dataset = get_best_features_dataset(best_result, feature_rankings, args.class_column)
    output_file = os.path.join(args.output, f'rfg_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_rfg)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished RFG Features Selection', 'info', logger_rfg)
    return True
