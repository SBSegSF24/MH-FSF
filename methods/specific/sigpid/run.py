import pandas as pd
import numpy  as np
import timeit
import argparse
import csv
import os, sys
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import logging
from tqdm import tqdm
from utils import *

def add_arguments(parser):
    """
    Adds arguments to the argument parser for SigPID.

    Parameters:
    parser (argparse.ArgumentParser): The argument parser instance.
    """
    parser = parser.add_argument_group('Arguments for SigPID')

def S_B(j):
    """
    Computes the sum of B for the jth column, normalized by the sizes of B and M.

    Parameters:
    j (int): The column index.

    Returns:
    float: The normalized sum of B for the jth column.
    """
    global B
    global M
    sigmaBij = B.sum(axis = 0, skipna = True)[j]
    sizeBj = B.shape[0]
    sizeMj = M.shape[0]
    return (sigmaBij/sizeBj)*sizeMj

def PRNR(j):
    """
    Computes the PRNR value for the jth column.

    Parameters:
    j (int): The column index.

    Returns:
    float: The PRNR value for the jth column.
    """
    sigmaMij = (M.sum(axis = 0, skipna = True)[j]) * 1.0
    S_Bj = S_B(j)
    r = (sigmaMij - S_Bj)/(sigmaMij + S_Bj) if sigmaMij > 0.0 and S_Bj > 0.0 else 0.0
    return r

def check_dirs(delete = False):
    """
    Checks if the MLDP directory exists and creates required subdirectories if not.

    Parameters:
    delete (bool): If True, deletes the MLDP directory.
    """
    import shutil
    root_path = os.getcwd()
    root_path = os.path.join(root_path, 'MLDP')
    if os.path.exists(root_path):
        shutil.rmtree('MLDP')
    if delete:
        return
    dirs = ['PRNR', 'SPR', 'PMAR']
    for dir in dirs:
        path = os.path.join(root_path, dir)
        os.makedirs(path)

def calculate_PRNR(dataset, filename, class_column):
    """
    Calculates the PRNR values for each permission in the dataset and writes them to a file.

    Parameters:
    dataset (DataFrame): The dataset.
    filename (str): The output file name.
    class_column (str): The name of the class column.
    """
    permissions = dataset.drop(columns = [class_column])
    with open(filename,'w', newline = '') as f:
        f_writer = csv.writer(f)
        for p in permissions:
            permission_PRNR_ranking = PRNR(p)
            if permission_PRNR_ranking != 0:
                f_writer.writerow([p, permission_PRNR_ranking])

def permission_list(filename, asc):
    """
    Reads a list of permissions and their ranks from a file and sorts them.

    Parameters:
    filename (str): The input file name.
    asc (bool): Sort in ascending order if True, descending if False.

    Returns:
    DataFrame: The sorted list of permissions.
    """
    colnames = ['permission', 'rank']
    list = pd.read_csv(filename, names = colnames)
    list = list.sort_values(by = ['rank'], ascending = asc)
    return list

def SVM(dataset, class_column):
    """
    Trains an SVM classifier on the dataset and returns the accuracy.

    Parameters:
    dataset (DataFrame): The dataset.
    class_column (str): The name of the class column.

    Returns:
    float: The accuracy of the SVM classifier.
    """
    state = np.random.randint(100)
    Y = dataset[class_column]
    X = dataset.drop([class_column], axis = 1)
    #split between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.3, random_state = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 1)

    svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)
    svm.fit(X_train,y_train)

    y_pred = svm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

def run_PMAR(dataset, prnr_malware, class_column):
    """
    Runs the PMAR algorithm on the dataset.

    Parameters:
    dataset (DataFrame): The dataset.
    prnr_malware (DataFrame): The PRNR malware permissions.
    class_column (str): The name of the class column.

    Returns:
    DataFrame: The reduced dataset after applying PMAR.
    """
    global logger_sigpid
    features = dataset.columns.values.tolist()
    data = dataset.drop([class_column], axis = 1)
    target = dataset[class_column]
    n_samples = dataset.shape[0] - 1
    n_features = data.shape[1]

    print_message('Mining Association Rules', 'info', logger_sigpid)
    records = list()
    for i in range(0, n_samples):
        if target[i] in [0, 1]:
            i_list = list()
            for j in range(0, n_features):
                if data.values[i][j] == 1:
                    i_list.append(features[j])
            records.append(i_list)

    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns = te.columns_)

    freq_items = apriori(df,
                        min_support = 0.1,
                        use_colnames = True,
                        max_len = 2,
                        verbose = 0)
    if not freq_items.empty:
        rules = association_rules(freq_items, metric = 'confidence', min_threshold = 0.965)
    else:
        rules = list()

    PMAR_dataset = dataset.copy()
    deleted_features = list()
    for i in range(0, len(rules)):
        ant = list(rules.loc[i,'antecedents'])[0]
        con = list(rules.loc[i,'consequents'])[0]
        rank_ant = prnr_malware.loc[(prnr_malware['permission'] == ant)].values[0,1]
        rank_con = prnr_malware.loc[(prnr_malware['permission'] == con)].values[0,1]
        to_delete = ant if rank_ant < rank_con else con
        if to_delete not in deleted_features:
            PMAR_dataset.drop([to_delete], axis = 1, inplace = True)
            deleted_features.append(to_delete)
    return PMAR_dataset

def drop_internet(dataset):
    """
    Drops internet-related permissions from the dataset.

    Parameters:
    dataset (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with internet-related permissions removed.
    """
    cols = list()
    to_drop = ['android.permission.INTERNET', 'INTERNET']
    features = dataset.columns.values.tolist()
    cols = list(set(features).intersection(to_drop))
    dataset_ = dataset.drop(columns = cols)
    return dataset_

def run(args, path, dataset):
    """
    Main function to run the SigPID features selection process.

    Parameters:
    args (Namespace): Command-line arguments.
    path (str): Path to the input dataset.
    dataset (DataFrame): The input dataset.

    Returns:
    bool: True if the process completes successfully.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_sigpid
    logger_sigpid = logging.getLogger('SigPID')
    logger_sigpid.setLevel(logging.INFO)

    check_dirs()
    reduced_dataset = drop_internet(dataset)

    global B
    global M
    print_message('Starting Features Selection', 'info', logger_sigpid)
    B = reduced_dataset[reduced_dataset[args.class_column] == 0]
    M = reduced_dataset[reduced_dataset[args.class_column] == 1]

    calculate_PRNR(B, 'MLDP/PRNR/PRNR_B_List.csv', args.class_column)
    calculate_PRNR(M, 'MLDP/PRNR/PRNR_M_List.csv', args.class_column)

    benigns_permissions = permission_list('MLDP/PRNR/PRNR_B_List.csv', True)
    malwares_permissions = permission_list('MLDP/PRNR/PRNR_M_List.csv' , False)
    num_permissions = reduced_dataset.shape[1] - 1 #CLASS

    print_message('PRNR Generating Subset of Permissions', 'info', logger_sigpid)
    counter = increment = 3
    while counter < num_permissions/2 + increment:
        malwares_head_perms = malwares_permissions['permission'].head(counter).values
        benigns_head_perms = benigns_permissions['permission'].head(counter).values
        subset_permissions = list(set(malwares_head_perms) | set(benigns_head_perms))
        subset_permissions.append(args.class_column)
        subset = reduced_dataset[subset_permissions]
        evaluated_ft = counter * 2
        evaluated_ft = num_permissions if evaluated_ft > num_permissions else evaluated_ft
        subset.to_csv(f'MLDP/PRNR/subset_{evaluated_ft}.csv', index = False)
        counter += increment

    counter = increment = 6
    best_PRNR_accuracy = 0.0
    best_PRNR_counter = 0

    with open('MLDP/PRNR/svm_results.csv', 'w', newline = '') as f:
        f_writer = csv.writer(f)
        print_message('Running PIS + PRNR', 'info', logger_sigpid)
        pbar = tqdm(range(num_permissions), disable = (logger_sigpid.getEffectiveLevel() > logging.INFO))
        while counter < num_permissions + increment:
            evaluated_ft = num_permissions if counter > num_permissions else counter
            pbar.set_description(f'SigPID (PIS + PRNR) With {evaluated_ft} Features')
            pbar.n = evaluated_ft
            dataset_df = pd.read_csv(f'MLDP/PRNR/subset_{evaluated_ft}.csv', encoding = 'utf8')
            accuracy = SVM(dataset_df, args.class_column)
            if accuracy > best_PRNR_accuracy:
                best_PRNR_accuracy = accuracy
                best_PRNR_counter = evaluated_ft
            f_writer.writerow([evaluated_ft, accuracy])
            counter += increment
        pbar.close()

    print_message(f'Best Accuracy: {(best_PRNR_accuracy * 100.0):.2f}. Number of Features: {best_PRNR_counter}', 'info', logger_sigpid)
    #SPR
    PRNR_dataset = pd.read_csv(f'MLDP/PRNR/subset_{best_PRNR_counter}.csv', encoding = 'utf8')
    PRNR_dataset.drop(columns = [args.class_column], inplace = True)

    #calculates the support of each permission
    supp = PRNR_dataset.sum(axis = 0)
    supp = supp.sort_values(ascending = False)

    print_message('SPR Generating Subset of Permissions', 'info', logger_sigpid)
    counter = increment = 5
    while counter < best_PRNR_counter + increment:
        subset_permissions = list(supp.head(counter).index)
        subset_permissions.append(args.class_column)
        subset = reduced_dataset[subset_permissions]
        evaluated_ft = best_PRNR_counter if counter > best_PRNR_counter else counter
        subset.to_csv(f'MLDP/SPR/subset_{evaluated_ft}.csv', index = False)
        counter += increment

    counter = increment = 5
    best_SPR_accuracy = best_PRNR_accuracy
    best_SPR_counter = best_PRNR_counter
    with open('MLDP/SPR/svm_results.csv','w', newline = '') as f:
        f_writer = csv.writer(f)
        pbar_spr = tqdm(range(best_PRNR_counter), disable = (logger_sigpid.getEffectiveLevel() > logging.INFO))
        while counter < best_PRNR_counter + increment:
            evaluated_ft = best_PRNR_counter if counter > best_PRNR_counter else counter
            pbar_spr.set_description(f'SigPID (PIS + SPR) With {evaluated_ft} Features')
            pbar_spr.n = evaluated_ft
            dataset_df = pd.read_csv(f'MLDP/SPR/subset_{evaluated_ft}.csv', encoding = 'utf8')
            accuracy = SVM(dataset_df, args.class_column)
            if accuracy >= 0.9 and evaluated_ft < best_SPR_counter:
                best_SPR_accuracy = accuracy
                best_SPR_counter = evaluated_ft
            f_writer.writerow([evaluated_ft, accuracy])
            counter += increment
        pbar_spr.close()

    print_message(f'SPR Pruning Point >> Best Accuracy: {(best_SPR_accuracy * 100.0):.2f}. Number of Features: {best_SPR_counter}', 'info', logger_sigpid)
    #PMAR
    SPR_dataset = pd.read_csv(f'MLDP/SPR/subset_{best_SPR_counter}.csv', encoding = 'utf8')
    reduced_dataset = run_PMAR(SPR_dataset, malwares_permissions, args.class_column)
    output_file = os.path.join(args.output, f'sigpid_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_sigpid)
    reduced_dataset.to_csv(output_file, index = False)

    check_dirs(delete = True)
    final_perms = reduced_dataset.shape[1] - 1
    num_permissions = dataset.shape[1] - 1
    pct = (1.0 - (final_perms/num_permissions)) * 100.0
    print_message(f'{num_permissions} to {final_perms} Features. Reduction of {pct:.2f}', 'info', logger_sigpid)
    print_message('Finished SigPID Features Selection', 'info', logger_sigpid)
    return True
