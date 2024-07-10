import math
import sys
import pandas as pd
from argparse import ArgumentParser
import os
import logging
from utils import *

def add_arguments(parser):
    """
    Add arguments specific to the Multi-Tiered feature selection script to the provided ArgumentParser object.

    Parameters:
        parser (ArgumentParser): ArgumentParser object to add arguments to.
    """
    parser = parser.add_argument_group('Arguments for Multi-Tiered')

def calculate_entropy(data, args):
    """
    Calculate the entropy of the target variable in the dataset.

    Parameters:
        data (DataFrame): Input dataset.
        args (Namespace): Parsed arguments containing class column name.

    Returns:
        float: Entropy value.
    """
    class_counts = data[args.class_column].value_counts()
    total_samples = len(data)
    entropy = 0.0
    for count in class_counts:
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_conditional_entropy(data, feature, args):
    """
    Calculate the conditional entropy of a feature in the dataset.

    Parameters:
        data (DataFrame): Input dataset.
        feature (str): Name of the feature column.
        args (Namespace): Parsed arguments containing class column name.

    Returns:
        float: Conditional entropy value.
    """
    unique_values = data[feature].unique()
    total_samples = len(data)
    conditional_entropy = 0.0
    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset, args)
        probability = len(subset) / total_samples
        conditional_entropy += probability * subset_entropy
    return conditional_entropy

def calculate_information_gain(data, feature, args):
    """
    Calculate the information gain of a feature in the dataset.

    Parameters:
        data (DataFrame): Input dataset.
        feature (str): Name of the feature column.
        args (Namespace): Parsed arguments containing class column name.

    Returns:
        float: Information gain value.
    """
    entropy = calculate_entropy(data, args)
    conditional_entropy = calculate_conditional_entropy(data, feature, args)
    information_gain = entropy - conditional_entropy
    return information_gain

def fib(feature):
    """
    Calculate the feature imbalance factor for a given feature in the benign class.

    Parameters:
        feature (str): Name of the feature column.

    Returns:
        float: Feature imbalance factor value.
    """
    global B
    return len(B[B[feature] != 0])/len(B)

def fim(feature):
    """
    Calculate the feature imbalance factor for a given feature in the malicious class.

    Parameters:
        feature (str): Name of the feature column.

    Returns:
        float: Feature imbalance factor value.
    """
    global M
    return len(M[M[feature] != 0])/len(M)

def score(feature):
    """
    Calculate the discrimination score for a given feature.

    Parameters:
        feature (str): Name of the feature column.

    Returns:
        float: Discrimination score value.
    """
    fb = fib(feature)
    fm = fim(feature)
    value = 1.0 - (min(fb, fm) / max(fb, fm))
    return value

def get_unique_values(df):
    """
    Generator function to yield unique values for each column in a dataframe.

    Parameters:
        df (DataFrame): Input dataframe.

    Yields:
        tuple: Column name and its unique values.
    """
    for column_name in df.columns:
        yield (column_name, df[column_name].unique())

def drop_irrelevant_columns(df):
    """
    Drop columns from the dataframe that have fewer than 2 unique values.

    Parameters:
        df (DataFrame): Input dataframe.

    Returns:
        DataFrame: DataFrame with irrelevant columns dropped.
    """
    irrelevant_columns = list()
    for column_name, unique_values in get_unique_values(df):
        if len(unique_values) < 2:
            irrelevant_columns.append(column_name)
    return df.drop(columns = irrelevant_columns)

def non_frequent_features(df, args, th = 0.1):
    """
    Identify non-frequent features based on a given threshold.

    Parameters:
        df (DataFrame): Input DataFrame.
        args (Namespace): Parsed arguments containing class column information.
        th (float): Threshold for frequency. Features below this threshold are considered non-frequent. Default is 0.1.

    Returns:
        list: List of non-frequent feature names.
    """
    df_len = len(df)
    non_frequent = list()
    X = df.drop(args.class_column, axis = 1)
    for ft in list(X.columns):
        count_nonzero = (X[ft] != 0).sum()
        frequency = count_nonzero/df_len
        if frequency < th:
            non_frequent.append(ft)
    return non_frequent

def features_to_drop(data, th = 0.1):
    """
    Identify features with values below a certain threshold and recommend them for dropping.

    Parameters:
        data (dict): Dictionary mapping feature names to their scores.
        th (float): Threshold value. Features with scores below this threshold will be recommended for dropping. Default is 0.1.

    Returns:
        list: List of feature names recommended for dropping.
    """
    max_value = max(data.values())
    min_value = max_value * th
    ft_to_drop = list()
    for ft, value in data.items():
        if value < min_value:
            ft_to_drop.append(ft)
    return ft_to_drop

def run(args, path, dataset):
    """
    Main function to execute the Multi-Tiered feature selection process and save the reduced dataset.

    Parameters:
        args (Namespace): Parsed arguments.
        path (str): Path to the dataset.
        dataset (DataFrame): Input dataset.

    Returns:
        bool: True if feature selection and dataset saving are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_mt
    global B
    global M
    logger_mt = logging.getLogger('MT')
    logger_mt.setLevel(logging.INFO)

    #dataset = get_dataset(args, ds, logger_mt)
    reduced_dataset = dataset.copy()
    print_message('Starting Features Selection', 'info', logger_mt)
    print_message(f'Number of Initial Features: {dataset.shape[1] - 1}', 'info', logger_mt)
    print_message(f'Non-Frequent Reduction', 'info', logger_mt)
    B = dataset[(dataset[args.class_column] == 0)]
    M = dataset[(dataset[args.class_column] == 1)]
    non_frequent_b_ft = non_frequent_features(B, args)
    non_frequent_m_ft = non_frequent_features(M, args)
    non_frequent_ft = list(set(non_frequent_b_ft).union(set(non_frequent_m_ft)))
    print_message(f'Number of Non-Frequent Features: {len(non_frequent_ft)}', 'info', logger_mt)
    reduced_dataset.drop(columns = non_frequent_ft, inplace = True)

    print_message('Feature Discrimination', 'info', logger_mt)
    ft_score = dict()
    for ft in list(reduced_dataset.columns):
        if ft != args.class_column:
            ft_score[ft] = score(ft)
    ft_to_drop = features_to_drop(ft_score)
    print_message(f'Number of Features to Drop: {len(ft_to_drop)}', 'info', logger_mt)
    reduced_dataset.drop(columns = ft_to_drop, inplace = True)

    print_message('Information Gain', 'info', logger_mt)
    ft_ig = dict()
    for ft in list(reduced_dataset.columns):
        if ft != args.class_column:
            ft_ig[ft] = calculate_information_gain(dataset, ft, args)
    ft_to_drop = features_to_drop(ft_ig)
    print_message(f'Number of Features to Drop: {len(ft_to_drop)}', 'info', logger_mt)
    reduced_dataset.drop(columns = ft_to_drop, inplace = True)
    output_file = os.path.join(args.output, f'mt_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_mt)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished Multi-Tiered (MT) Features Selection', 'info', logger_mt)
    return True
