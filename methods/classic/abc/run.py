import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
from utils import *
from termcolor import colored

class ArtificialBeeColony:
    def __init__(self, n_bees, n_iterations, limit, n_features):
        """
        Initialize the Artificial Bee Colony algorithm for feature selection.

        Parameters:
        - n_bees (int): Number of bees (solutions).
        - n_iterations (int): Number of iterations.
        - limit (int): Abandonment limit for solutions.
        - n_features (int): Number of features in the dataset.
        """
        self.n_bees = n_bees
        self.n_iterations = n_iterations
        self.limit = limit
        self.n_features = n_features
        self.food_sources = np.random.randint(2, size = (n_bees, n_features))
        self.fitness = np.zeros(n_bees)
        self.trial_counters = np.zeros(n_bees)

    def evaluate_fitness(self, data, target):
        """
        Evaluate fitness for each bee's solution.

        Parameters:
        - data (numpy.ndarray): Input features data.
        - target (numpy.ndarray): Target labels.

        Computes accuracy using Random Forest Classifier for each solution.
        """
        for i in range(self.n_bees):
            features = self.food_sources[i]
            if np.sum(features) == 0:
                self.fitness[i] = 0
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    data[:, features == 1], target, test_size = 0.3, random_state = 0)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                self.fitness[i] = accuracy_score(y_test, predictions)

    def employed_bees_phase(self, data, target):
        """
        Employed bees phase: attempt to improve solutions.

        Parameters:
        - data (numpy.ndarray): Input features data.
        - target (numpy.ndarray): Target labels.

        Bees mutate solutions and update if fitness improves.
        """
        for i in range(self.n_bees):
            new_solution = self.mutate_solution(self.food_sources[i])
            new_fitness = self.evaluate_single_solution(new_solution, data, target)
            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1

    def onlooker_bees_phase(self, data, target):
        """
        Onlooker bees phase: probabilistic solution improvement.

        Parameters:
        - data (numpy.ndarray): Input features data.
        - target (numpy.ndarray): Target labels.

        Bees select solutions based on fitness and attempt to improve them.
        """
        max_fitness = np.max(self.fitness)
        for i in range(self.n_bees):
            if np.random.rand() < self.fitness[i] / (max_fitness + 1e-6):
                new_solution = self.mutate_solution(self.food_sources[i])
                new_fitness = self.evaluate_single_solution(new_solution, data, target)
                if new_fitness > self.fitness[i]:
                    self.food_sources[i] = new_solution
                    self.fitness[i] = new_fitness
                    self.trial_counters[i] = 0
                else:
                    self.trial_counters[i] += 1

    def scout_bees_phase(self):
        """
        Scout bees phase: abandon solutions that exceed limit.

        Replaces solutions that exceed the abandonment limit.
        """
        for i in range(self.n_bees):
            if self.trial_counters[i] > self.limit:
                self.food_sources[i] = np.random.randint(2, size = self.n_features)
                self.fitness[i] = 0
                self.trial_counters[i] = 0

    def mutate_solution(self, solution):
        """
        Mutate a solution by flipping a feature.

        Parameters:
        - solution (numpy.ndarray): Binary solution (0s and 1s).

        Returns:
        - numpy.ndarray: Mutated solution.
        """
        new_solution = solution.copy()
        feature_to_change = np.random.randint(self.n_features)
        new_solution[feature_to_change] = 1 - new_solution[feature_to_change]
        return new_solution

    def evaluate_single_solution(self, solution, data, target):
        """
        Evaluate fitness of a single solution.

        Parameters:
        - solution (numpy.ndarray): Binary solution (0s and 1s).
        - data (numpy.ndarray): Input features data.
        - target (numpy.ndarray): Target labels.

        Returns:
        - float: Accuracy score of the solution.
        """
        if np.sum(solution) == 0:
            return 0
        X_train, X_test, y_train, y_test = train_test_split(
            data[:, solution == 1], target, test_size = 0.3, random_state = 0)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)

    def optimize(self, data, target):
        """
        Optimize feature selection using ABC algorithm.

        Parameters:
        - data (numpy.ndarray): Input features data.
        - target (numpy.ndarray): Target labels.

        Returns:
        - numpy.ndarray: Best set of features.
        """
        self.evaluate_fitness(data, target)
        for i in range(self.n_iterations):
            self.employed_bees_phase(data, target)
            self.onlooker_bees_phase(data, target)
            self.scout_bees_phase()
        best_index = np.argmax(self.fitness)
        return self.food_sources[best_index]

def add_arguments(parser):
    """
    Add arguments for ABC (Artificial Bee Colony) algorithm.

    Parameters:
    - parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = parser.add_argument_group('Arguments for ABC')
    parser.add_argument(
        '-nb', '--n_bees', metavar = 'INT', type = int, default = 20,
        help = 'Number of Bees. Default: 20')
    parser.add_argument(
        '-l', '--limit', metavar = 'INT', type = int, default = 3,
        help = 'Abandonment Limit. Default: 3')
    parser.add_argument(
        '-nit', '--n_iterations', metavar = 'INT', type = int, default = 3,
        help = 'Number of Iterations. Default: 3')

def run(args, path, dataset):
    """
    Run the ABC algorithm for feature selection.

    Parameters:
    - args (argparse.Namespace): Arguments parsed from command line.
    - path (str): File path of the dataset.
    - dataset (pd.DataFrame): Input dataset.

    Returns:
    - bool: True if feature selection and saving the reduced dataset are successful.
    """
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_abc
    logger_abc = logging.getLogger('ABC')
    logger_abc.setLevel(logging.INFO)

    X, y = get_X_y(args, dataset, logger_abc)
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    print_message('Starting Features Selection', 'info', logger_abc)
    abc = ArtificialBeeColony(args.n_bees, args.n_iterations, args.limit, X.shape[1])
    best_features = abc.optimize(X_np, y_np)
    selected_column_index = np.where(best_features == 1)[0]
    selected_columns = dataset.columns[selected_column_index].tolist()
    selected_columns.append(args.class_column)
    reduced_dataset = dataset[selected_columns]
    output_file = os.path.join(args.output, f'abc_{os.path.basename(path)}')
    print_message('Saving Reduced Dataset', 'info', logger_abc)
    reduced_dataset.to_csv(output_file, index = False)
    print_message('Finished ABC Features Selection', 'info', logger_abc)
    return True
