'''
Import libraries
'''

def add_arguments(parser):
    """
    Add arguments specific to method to the argparse.ArgumentParser object.

    Parameters:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for SemiDroid')

'''
Others functions, if necessary
'''

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


    """
    To save reduced dataset
    """
    method_id = 'use the same directory name' #use the same directory name
    output_file = os.path.join(args.output, f'{method_id}_{os.path.basename(path)}')
    reduced_dataset.to_csv(output_file, index = False)
    return True
