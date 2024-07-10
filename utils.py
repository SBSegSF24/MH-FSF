import argparse
import pandas as pd
import os
import logging
from termcolor import colored

def print_message(info, type, logger):
    """
    Prints a colored message to the console and logs it.

    Parameters:
    info (str): The message to print and log.
    type (str): The type of message ('warn', 'info', 'except', 'error').
    logger (Logger): The logger instance to use.
    """
    if type == 'warn':
        message = colored(f'{info}', 'yellow')
        logger.warning(message)
    elif type == 'info':
        message = colored(f'{info}', 'green')
        logger.info(message)
    elif type == 'except':
        message = colored(f'{str(info)}', 'red')
        logger.exception(message)
    elif type == 'error':
        message = colored(f'{str(info)}', 'red')
        logger.error(message)
    else:
        logger.info(info)

def get_dataset(args, dataset, logger):
    """
    Loads a dataset from a CSV file.

    Parameters:
    args (Namespace): Command-line arguments.
    dataset (str): Path to the dataset file.
    logger (Logger): The logger instance to use.

    Returns:
    DataFrame: Loaded dataset as a pandas DataFrame, or None if loading fails.
    """
    if not os.path.isfile(dataset):
        print_message(f'Dataset File {dataset} Not Found', 'error', logger)
        return None
    try:
        dataset_df = pd.read_csv(dataset, low_memory = False)
    except Exception as e:
        print_message(e, 'except', logger)
        return None
    return dataset_df

def get_X_y(args, dataset, logger):
    """
    Splits the dataset into features (X) and labels (y).

    Parameters:
    args (Namespace): Command-line arguments.
    dataset (DataFrame): The loaded dataset.
    logger (Logger): The logger instance to use.

    Returns:
    Tuple: A tuple containing the feature set (X) and the label set (y).
    """
    if args.class_column not in dataset.columns:
        message = f'Dataset Does Not Have a Column Called "{args.class_column}"'
        print_message(message, 'error', logger)
        exit(1)
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def check_directory(dir):
    """
    Checks if a directory exists and creates it if it does not.

    Parameters:
    dir (str): The directory path to check and create if necessary.
    """
    root_path = os.getcwd()
    dir_path = os.path.join(root_path, dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir)

def get_dir_list(dir_path):
    """
    Gets a list of all subdirectories in a given directory, excluding '__pycache__'.

    Parameters:
    dir_path (str): The path of the directory to scan.

    Returns:
    list: A list of subdirectory names.
    """
    l = list()
    for it in os.scandir(dir_path):
        if it.is_dir():
            l.append(it.name)
    if '__pycache__' in l:
        l.remove('__pycache__')
    return l

def find_constant_features(df):
    """
    Identifies columns in the DataFrame that have only one unique value.

    Parameters:
    df (DataFrame): The DataFrame to scan for constant features.

    Returns:
    list: A list of column names that are constant.
    """
    constant_features = list()
    for column in df.columns:
        if df[column].nunique() == 1:
            constant_features.append(column)
    return constant_features
