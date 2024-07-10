import pandas as pd
import numpy as np
import sys
import argparse
from sklearn.feature_selection import chi2
from scipy.stats import entropy
from collections import Counter
from math import log
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import logging
from utils import *

def add_arguments(parser):
    """
    Add arguments specific to SemiDroid to the argparse.ArgumentParser object.

    Parameters:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for SemiDroid')

def _Ex_a_v_(Ex, a, v, nan = True):
    """
    Filter elements of list Ex based on values in list a and value v.

    Parameters:
        Ex (list): List of elements to filter.
        a (list): List of values to match against Ex.
        v (float): Value to match against elements in a.
        nan (bool, optional): Whether to include NaN values. Defaults to True.

    Returns:
        list: Filtered elements from Ex that match the condition.
    """
    if nan:
        return [x for x, t in zip(Ex, a) if (isinstance(t, float)
                                        and isinstance(v, float)
                                        and math.isnan(t)
                                        and math.isnan(v))
                                        or t == v]
    else:
        return [x for x, t in zip(Ex, a) if t == v]

def intrinsic_value(Ex, a, nan = True):
    """
    Calculate the intrinsic value based on the frequency of values in list a.

    Parameters:
        Ex (list): List of elements.
        a (list): List of values to calculate intrinsic value.
        nan (bool, optional): Whether to include NaN values. Defaults to True.

    Returns:
        float: Intrinsic value calculated based on the frequency of values in a.
    """
    sum_v = 0
    for v in set(a):
        Ex_a_v = _Ex_a_v_(Ex, a, v, nan)
        if len(Ex_a_v) != 0:
            return (len(Ex_a_v) / len(Ex)) * (log(len(Ex_a_v) / len(Ex), 2))

def get_subset(X, y, data, args):
    """
    Select a subset of features from dataframe X based on given data and arguments.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        data (DataFrame): Dataframe containing feature scores.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features and target variable y.
    """
    data.sort_values(by = ['score'], ascending = False, inplace = True)
    n = int(X.shape[1] * args.threshold + 1)
    selected_features = data.feature[:n]
    sub_df = X.loc[:,selected_features]
    sub_df[args.class_column] = y
    return sub_df

def chi_squared(X, y, args):
    """
    Perform chi-squared feature selection on data X and target y.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features based on chi-squared scores.
    """
    feature_names = X.columns
    chi2score, pval = chi2(X, y)
    data = pd.DataFrame({'feature': feature_names, 'score': chi2score})
    return get_subset(X, y, data, args)

def i_gain(Ex, a, nan = True):
    """
    Calculate the information gain for attribute a in dataset Ex.

    Parameters:
        Ex (list): Dataset to calculate information gain.
        a (Series): Attribute series to calculate information gain.
        nan (bool, optional): Whether to include NaN values. Defaults to True.

    Returns:
        float: Information gain for attribute a in dataset Ex.
    """
    H_Ex = entropy(list(Counter(Ex).values()))
    sum_v = 0
    for v in set(a):
        Ex_a_v = _Ex_a_v_(Ex, a, v, nan)
        sum_v += (len(Ex_a_v) / len(Ex)) *\
            (entropy(list(Counter(Ex_a_v).values())))
    result = H_Ex - sum_v
    return result

def info_gain(X, y, args):
    """
    Perform information gain feature selection on data X and target y.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features based on information gain scores.
    """
    values = list()
    for i in X.columns:
        values.append((i, i_gain(i, X)))
    data = pd.DataFrame(values, columns = ['feature', 'score'])
    return get_subset(X, y, data, args)

def g_ratio(Ex, a, nan = True):
    """
    Calculate the gain ratio for attribute a in dataset Ex.

    Parameters:
        Ex (list): Dataset to calculate gain ratio.
        a (Series): Attribute series to calculate gain ratio.
        nan (bool, optional): Whether to include NaN values. Defaults to True.

    Returns:
        float: Gain ratio for attribute a in dataset Ex.
    """
    result = i_gain(Ex, a) /(-intrinsic_value(Ex, a))
    return result

def gain_ratio(X, y, args):
    """
    Perform gain ratio feature selection on data X and target y.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features based on gain ratio scores.
    """
    values = list()
    for i in X.columns:
        values.append((i, g_ratio(i, X)))
    data = pd.DataFrame(values, columns = ['feature', 'score'])
    return get_subset(X, y, data, args)

def one_r(X, y, args):
    """
    Perform One-R feature selection on data X and target y.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features based on One-R scores.
    """
    feature_names = X.columns
    result_list = list()
    summary_dict = dict()
    for feature_name in feature_names:
        summary_dict[feature_name] = {}
        join_data = pd.concat([X[feature_name], y], axis=1)
        freq_table = pd.crosstab(join_data[feature_name], join_data[y.name])
        summary = freq_table.idxmax(axis = 1)
        summary_dict[feature_name] = dict(summary)
        counts = 0
        for idx, row in join_data.iterrows():
            if row[y.name] == summary[row[feature_name]]:
                counts += 1
        accuracy = counts/len(y)
        result_feature = {'feature': feature_name, 'score': accuracy}
        result_list.append(result_feature)
    data = pd.DataFrame(result_list)
    return get_subset(X, y, data, args)

def pca_analysis(X, y, args):
    """
    Perform Principal Component Analysis (PCA) feature selection on data X and target y.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features based on PCA scores.
    """
    values = X.values
    x = StandardScaler().fit_transform(X)
    principal = PCA()
    principal.fit_transform(x)
    eigenvalues = principal.explained_variance_
    feature_names = X.columns
    data = pd.DataFrame({'feature': feature_names, 'score': eigenvalues})
    data = data[data['score'] > 1.0]
    return get_subset(X, y, data, args)

def logistic_regression(X, y, args):
    """
    Perform Logistic Regression feature selection on data X and target y.

    Parameters:
        X (DataFrame): Input features dataframe.
        y (Series): Target variable series.
        args (Namespace): Parsed arguments.

    Returns:
        DataFrame: Subset of X dataframe with selected features based on Logistic Regression coefficients.
    """
    feature_names = X.columns
    model = LogisticRegression(max_iter = 1000, solver = 'saga', random_state = 0).fit(X, y)
    score = model.coef_[0]
    data = pd.DataFrame({'feature': feature_names, 'score': score})
    data = data[data['score'] > 0.05]
    return get_subset(X, y, data, args)

def random_forest(dataset, args):
    """
    Train a Random Forest classifier on dataset and return accuracy score.

    Parameters:
        dataset (DataFrame): Input dataset with features and target variable.
        args (Namespace): Parsed arguments.

    Returns:
        float: Accuracy score of the Random Forest classifier on test data.
    """
    X = dataset.drop(args.class_column, axis = 1)
    y = dataset[args.class_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def run(args, path, dataset):
    """
    Perform feature selection using multiple methods and save the reduced dataset.

    Parameters:
        args (Namespace): Parsed arguments.
        path (str): Path to the dataset.
        dataset (DataFrame): Input dataset.

    Returns:
        bool: True if feature selection and dataset saving are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_semi
    logger_semi = logging.getLogger('SemiDroid')
    logger_semi.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_semi)
    print_message('Starting Features Selection', 'info', logger_semi)
    best_accuracy = 0.0
    reduced_dataset = pd.DataFrame()

    function_list = ['chi_squared', 'info_gain', 'gain_ratio', 'one_r', 'logistic_regression', 'pca_analysis']
    for function in function_list:
        function_real = globals()[function]
        print_message(f'Testing Dataset With {function}', 'info', logger_semi)
        subset = function_real(X, y, args)
        acc = random_forest(subset, args)
        print_message(f'Subset Accuracy: {acc * 100.0:.2f}', 'info', logger_semi)
        if acc > best_accuracy:
            best_accuracy = acc
            reduced_dataset = subset

    output_file = os.path.join(args.output, f'semidroid_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_semi)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished SemiDroid Features Selection', 'info', logger_semi)
    return True
