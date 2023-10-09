from explores import concatenate_reversal_datasets
from itertools import product
from metadata_queries.search import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from utils import create_train_test_sets
import pandas as pd


variable_coupling_neurons = [
        ["AVKL", "AVKR"] + ["RIR"],
        ["RIH"],
        ["URBR", "URBL"],
        ["RMDDL"],
        ["IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR", "IL1DL", "IL1DR",
            "IL1L", "IL1R"]
]

class BaseModel:

    def __init__(self):
        self.model = None

    def average_r2_cv(self, dataset_df):
        train_ratio = 0.8
        num_train_test_splits = 5
        num_augmentations = 0
        noise_multiplier = 1
        noise_ds_name = "2022-01-07-03/2022-01-07-03_F20.csv"
        num_label = 1

        train_test_sets = create_train_test_sets(
                           dataset_df,
                           noise_ds_name,
                           train_ratio,
                           num_train_test_splits,
                           num_augmentations,
                           noise_multiplier,
                           num_label)
        neural_trains = train_test_sets["neural_trains"]
        label_trains = train_test_sets["label_trains"]
        neural_tests = train_test_sets["neural_tests"]
        label_tests = train_test_sets["label_tests"]
        print(neural_trains[0].shape)

        for i in range(num_train_test_splits):
            model._fit(neural_trains[i], label_trains[i])
            predicted_label = model._predict(neural_tests[i])
            r2 = r2_score(label_tests[i], predicted_label)
            print(f'r2: {r2}')
            aggregate_r2_score += r2

        return aggregate_r2_score / num_train_test_splits

    def _fit(self, X, y):
        if self.model is None:
            raise NotImplementedError("Subclass must define the model.")
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)


class LinearModel(BaseModel):

    def __init__(self):
        self.model = LinearRegression()


class PolynormialModel(BaseModel):

    def __init__(self, degree):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())


def combine(neurons):

    combinations = list(product(*neurons))
    print(len(combinations))

    for combination in combinations:
        #print(list(combination))
        datasets = find_common_datasets_for_neurons(list(combination))
        print(f"{combination}: {list(datasets.keys())}")


def build_heuristics(datasets, neurons, num_behavior):

    dataset_df = concatenate_reversal_datasets(datasets,
                    neurons,
                    normalization=10,
                    linearize=True,
                    export_csv=False)
    linear_model = LinearModel()
    avg_r2_linear = linear_model.average_r2_cv(dataset_df)
    print(f"avg_r2_linear: {avg_r2_linear}")


if __name__ == "__main__":
    #combine(variable_coupling_neurons)
    build_heuristics(datasets, neurons, num_behavior)

