import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import *
import seaborn as sns
from math import pi

def graph_metrics(data_df, args, method, dataset):
    """
    Generates a bar graph to visualize various performance metrics.

    Parameters:
    data_df (DataFrame): DataFrame containing the performance metrics.
    args (Namespace): Command-line arguments.
    method (str): Feature selection method.
    dataset (str): Name of the dataset file.
    """
    data_df['model'] = data_df['model'].str.upper()
    metrics = ['accuracy','precision','recall','f1','roc_auc', 'mcc']
    data_df[metrics] *= 100
    melted_df = pd.melt(data_df, id_vars = 'model', value_vars = metrics)
    fig, ax = plt.subplots(figsize = (10, 6))
    sns.barplot(x = 'variable', y = 'value', hue = 'model', data = melted_df, ax = ax, edgecolor = 'white', linewidth = 2)
    # add annotations to show the values on each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
            xy = (p.get_x() + p.get_width() / 2, p.get_y() + 10),
            ha = 'center', va = 'bottom',
            rotation = 90, color = 'white', fontsize = 12, fontweight = 'bold')
    ax.set_ylim(0, 110)
    metrics_label = ['Accuracy','Precision','Recall','F1Score','RoC', 'MCC']
    positions = range(len(metrics_label))
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics_label)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor = (0.5, 1.), ncol = 3)
    ax.set_title(f'Results for {method} With Dataset in {os.path.basename(dataset)}')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values (%)')
    path_graph_file = os.path.join(args.output, f'metrics_{method}_{os.path.basename(dataset).replace(".csv", ".pdf")}')
    plt.savefig(path_graph_file)
    plt.close()

def graph_class(data_df, args, method, dataset, label_threshold = 2.5):
    """
    Generates a stacked bar graph to visualize classification results.

    Parameters:
    data_df (DataFrame): DataFrame containing the classification results.
    args (Namespace): Command-line arguments.
    method (str): Feature selection method.
    dataset (str): Name of the dataset file.
    label_threshold (float): Threshold for displaying labels on bars. Default is 2.5.
    """
    sns.set(style = 'white')
    models_index = list(data_df['model'].str.upper())
    classification_dict = dict()
    classification_list = ['TP', 'FP', 'TN', 'FN']
    for classification in classification_list:
        classification_dict[classification] = list(data_df[classification.lower()])

    df = pd.DataFrame(classification_dict, index=models_index)
    stacked_data = df.apply(lambda x: x * 100 / sum(x), axis = 1)

    plt.figure(figsize = (10, 5))
    bottom = pd.Series([0] * len(stacked_data), index = stacked_data.index)
    colors = sns.color_palette(n_colors = len(classification_list))#'tab10', len(classification_list))

    for i, classification in enumerate(classification_list):
        sns.barplot(x = stacked_data[classification], y = stacked_data.index, left = bottom,
                    color = colors[i], label = classification)
        bottom += stacked_data[classification]

    plt.xlabel('Values (%)')
    plt.ylabel('Model')
    plt.ylim(-1, len(models_index))
    plt.legend(ncol = len(classification_list), loc = 'upper center')
    plt.title(f'Classification to {method} With Dataset in {os.path.basename(dataset)}')

    # Add labels only if they are above the threshold
    bottom = pd.Series([0] * len(stacked_data), index = stacked_data.index)
    for i, classification in enumerate(classification_list):
        for j, value in enumerate(stacked_data[classification]):
            if value > label_threshold:
                plt.text(bottom[stacked_data.index[j]] + value / 2, j, f'{value:.2f}', ha = 'center', va = 'center',
                         color = 'black', weight = 'bold', rotation = 90)
        bottom += stacked_data[classification]
    path_graph_file = os.path.join(args.output, f'class_{method}_{os.path.basename(dataset).replace(".csv", ".pdf")}')
    plt.savefig(path_graph_file)
    plt.close()

def graph_radar(data_df, args, method, dataset):
    """
    Generates a radar chart to visualize performance metrics.

    Parameters:
    data_df (DataFrame): DataFrame containing the performance metrics.
    args (Namespace): Command-line arguments.
    method (str): Feature selection method.
    dataset (str): Name of the dataset file.
    """
    df = data_df[['model','accuracy', 'recall', 'mcc']]
    df.columns = ['Model', 'Accuracy', 'Recall', 'MCC']
    num_models = df.shape[0]
    categories = list(df.columns[1:])
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize = (10, 8), subplot_kw = dict(polar = True))
    for i in range(num_models):
        values = df.loc[i].drop('Model').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth = 2, linestyle = 'solid', label = df['Model'][i])
        ax.fill(angles, values, alpha = 0.25)

    # Add Tags
    plt.xticks(angles[:-1], categories)
    plt.legend(loc = 'upper right', bbox_to_anchor = (0.1, 0.1))
    plt.title(f'Comparison to {method} With Dataset in {os.path.basename(dataset)}')
    path_graph_file = os.path.join(args.output, f'radar_{method}_{os.path.basename(dataset).replace(".csv", ".pdf")}')
    plt.savefig(path_graph_file)
    plt.close()
