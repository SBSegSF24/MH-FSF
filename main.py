#!/usr/bin/python3
import argparse
from termcolor import colored, cprint
from evaluation import *
import argparse
import sys
import pandas as pd
import seaborn as sns
import logging
import os
from importlib import import_module
from utils import *
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def float_range(mini,maxi):
    """
    Creates a function to check if a float argument is within a specific range.

    Parameters:
    mini (float): Minimum value of the range (exclusive).
    maxi (float): Maximum value of the range (exclusive).

    Returns:
    function: A function that checks if a float is within the given range.
    """
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a Floating Point Number")
        if f <= mini or f >= maxi:
            raise argparse.ArgumentTypeError("Must be > " + str(mini) + " and < " + str(maxi))
        return f
    return float_range_checker

class DefaultHelpParser(argparse.ArgumentParser):
    """
    Custom argument parser that prints help message on error and logs the error.
    """
    def error(self, message):
        global logger
        self.print_help()
        msg = colored(message, 'red')
        logger.error(msg)
        sys.exit(2)

def load_methods_args(parser, args):
    """
    Loads arguments for the selected features selection methods.

    Parameters:
    parser (argparse.ArgumentParser): The argument parser.
    args (argparse.Namespace): Parsed command line arguments.
    """
    global methods_path
    global methods_types
    global methods_dict

    selected_methods = args.fs_methods
    selected_types = methods_types
    if args.fs_types:
        selected_types = args.fs_types

    for type in selected_types:
        methods = methods_dict[type]
        for mth in methods:
            if not selected_methods or mth in selected_methods:
                module = '.'.join([methods_path, type, mth, 'run'])
                model_instance = import_module(module)
                model_instance.add_arguments(parser)

def modify_choices(parser, dest, choices):
    """
    Modifies the choices for a specific argument in the parser.

    Parameters:
    parser (argparse.ArgumentParser): The argument parser.
    dest (str): The argument destination to modify.
    choices (list): The new choices for the argument.
    """
    for action in parser._actions:
        if action.dest == dest:
            action.choices = choices
            action.help += '. Choices: ' + str(choices)
            return
    else:
        raise AssertionError('argument {} not found'.format(dest))

def parse_args(argv):
    """
    Parses command line arguments and returns them.

    Parameters:
    argv (list): Command line arguments.

    Returns:
    argparse.Namespace: Parsed arguments.
    """
    global methods_types
    global methods_dict
    global ml_models

    action_base_parser = argparse.ArgumentParser(add_help = False)
    action_base_group = action_base_parser.add_mutually_exclusive_group(required = True)
    action_base_group.add_argument(
        '--fs-types', nargs = '+', metavar = 'TYPE',
        help = 'Types of Feature Selection (FS) Methods. Choices: ' + str(methods_types),
        choices = methods_types, type = str)
    action_base_group.add_argument(
        '--all-fs-types', help = f'All Types of Feature Selection (FS) Methods',
        action = 'store_true')

    parser = DefaultHelpParser(formatter_class = argparse.RawTextHelpFormatter)
    parser._optionals.title = 'Optional Arguments'
    action_subparser = parser.add_subparsers(title = 'Available Actions', dest = 'action', metavar = 'ACTION', required = True)
    action_list = action_subparser.add_parser(
        'list', help = 'List Features Selection (FS) Methods',
        parents = [action_base_parser])
    action_list._optionals.title = 'Optional Arguments'
    action_run = action_subparser.add_parser(
        'run', help = 'Run Features Selection (FS) Methods',
        parents = [action_base_parser])
    action_run._optionals.title = 'Optional Arguments'
    args_, _ = parser.parse_known_args()

    available_methods = list()
    selected_types = methods_types
    if args_.action == 'run':
        if args_.fs_types:
            selected_types = args_.fs_types
        for type in selected_types:
            available_methods += methods_dict[type]

    action_run_group = action_run.add_mutually_exclusive_group(required = True)
    action_run_group.add_argument(
        '--fs-methods', nargs = '+', metavar = 'METHOD',
        help = 'Run Selected Methods. Choices: ' + str(available_methods),
        choices = available_methods, type = str)
    action_run_group.add_argument(
        '--all-fs-methods', help = f'Run ALL Available Methods',
        action = 'store_true')
    action_run.add_argument(
        '-d', '--datasets', nargs = '+', metavar = 'DATASET',
        help = 'One or More Datasets (csv Files). For All Datasets in Directory Use: [DIR_PATH]/*.csv',
        type = str,  required = True)
    action_run.add_argument('-th','--threshold', metavar = 'FLOAT',
        help = 'Percent of Features to be Selected (Ranking Methods). Default: 0.5',
        type = float_range(0.0, 1.0), default = 0.5)
    action_run.add_argument('-c', '--class-column', type = str, default = 'class', metavar = 'CLASS_COLUMN',
        help = 'Name of Class Column. Default: class')
    action_run.add_argument(
        '--parallelize', help = 'Parallel Execution',
        action = 'store_true')
    action_run.add_argument(
        '--output', help = 'Output File Directory. Default: results',
        type = str, default = 'results')
    ml_run_group = action_run.add_mutually_exclusive_group(required = True)
    ml_run_group.add_argument(
        '--ml-models', nargs = '+', metavar = 'MODEL',
        help = 'ML Model for Evaluation of Datasets Resulting from Features Selection (FS). Choices: ' + str(ml_models),
        choices = ml_models, type = str)
    ml_run_group.add_argument(
        '--all-ml-models', help = f'Run ALL Available ML Models',
        action = 'store_true')
    args_, _ = parser.parse_known_args()

    if args_.action == 'run':
        load_methods_args(action_run, args_)

    args = parser.parse_args(argv)
    return args

def list_methods(selected_methods_types):
    """
    Lists all features selection methods for the selected types.

    Parameters:
    selected_methods_types (list): List of selected features selection types.
    """
    for type in selected_methods_types:
        dir_path = os.path.join(methods_path, type)
        methods_in_dir = get_dir_list(dir_path)
        f = open(os.path.join(dir_path, "about.desc"), "r")
        method_desc = f.read()
        print(colored("\n>>> " + method_desc, 'green'))
        for i in methods_in_dir:
            f = open(os.path.join(dir_path, i, "about.desc"), "r")
            method_desc = f.read()
            print(colored("\t" + method_desc, 'yellow'))
    exit(1)

def get_methods():
    """
    Retrieves all features selection methods from the methods directory.

    Returns:
    dict: Dictionary with features selection methods categorized by type.
    """
    global methods_path
    global methods_types
    d = dict()
    for type in methods_types:
        methods_type_path = os.path.join(methods_path, type)
        dir_list = get_dir_list(methods_type_path)
        d[type] = dir_list
    return d

def parallelize_func(func, parameters, cores = cpu_count()):
    """
    Parallelizes the execution of a function across multiple cores.

    Parameters:
    func (function): The function to parallelize.
    parameters (list): List of parameters for the function.
    cores (int): Number of cores to use. Default is the number of CPU cores.

    Returns:
    list: List of results from the parallelized function.
    """
    with Pool(cores) as pool:
        results = list(pool.imap(func, parameters))
    return results

def run_fs_method(args, path, dataset, method_to_exec):
    """
    Runs a features selection method on a dataset.

    Parameters:
    args (argparse.Namespace): Parsed command line arguments.
    path (str): Path to the dataset.
    dataset (pandas.DataFrame): The dataset to process.
    method_to_exec (tuple): Tuple containing the type and name of the method to execute.

    Returns:
    bool: True if the method executed successfully, False otherwise.
    """
    global logger
    global methods_path
    global ml_models
    type, method = method_to_exec
    msg = f"Running Method {colored(method, 'blue')} to Dataset {colored(os.path.basename(path), 'blue')}"
    logger.info(msg)
    module = '.'.join([methods_path, type, method, 'run'])
    model_instance = import_module(module)
    try:
        model_instance.run(args, path, dataset)
        selected_ml_models = args.ml_models if args.ml_models else ml_models
        run_ml_models(args, selected_ml_models, method, path)
    except Exception as e:
        msg = f'Error Executing Method {method} in {os.path.basename(path)}: {e}'
        logger.exception(msg)
        return False
    return True

if __name__ == '__main__':
    global logger
    global methods_path
    global methods_types
    global methods_dict
    global ml_models
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('MHFSF')
    logger.setLevel(logging.INFO)
    methods_path = 'methods'
    methods_types = get_dir_list(methods_path)
    methods_dict = get_methods()
    check_files(methods_dict, logger)
    ml_models = list(available_ml_models.keys())
    args = parse_args(sys.argv[1:])

    if args.action == 'list':
        if args.fs_types:
            list_methods(args.fs_types)
        elif args.all_fs_types:
            list_methods(methods_types)

    datasets = args.datasets
    check_directory(args.output)

    selected_methods = args.fs_methods or None
    selected_types = args.fs_types or methods_types
    methods_to_exec = list()
    for type in selected_types:
        methods = methods_dict[type]
        for method in methods:
            if not selected_methods or method in selected_methods:
                methods_to_exec.append((type, method))

    for path in datasets:
        msg = f"Loading Dataset From {colored(os.path.basename(path), 'blue')}"
        print_message(msg, '', logger)
        dataset = get_dataset(args, path, logger)
        if dataset is None:
            continue
        try:
            if args.parallelize:
                output_e = parallelize_func(partial(run_fs_method, args, path, dataset), methods_to_exec)
                #print(output_e)
            else:
                for method_to_exec in methods_to_exec:
                    run_fs_method(args, path, dataset, method_to_exec)
        except Exception as e:
            msg = f'Error in Execution: {e}'
            logger.exception(msg)
