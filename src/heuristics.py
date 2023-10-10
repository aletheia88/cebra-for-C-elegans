from explores import concatenate_reversal_datasets
from itertools import product
from metadata_queries.search import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
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

        aggregate_r2_score = 0
        for i in range(num_train_test_splits):
            self._fit(neural_trains[i], label_trains[i])
            predicted_label = self._predict(neural_tests[i])
            r2 = r2_score(label_tests[i], predicted_label)
            print(f'r2: {r2}')
            aggregate_r2_score += r2

        return aggregate_r2_score / num_train_test_splits

    def _fit(self, X, y):
        if self.model is None:
            raise NotImplementedError("Subclass must define the model.")
        self.model.fit(X, y.ravel())

    def _predict(self, X):
        return self.model.predict(X)


class LinearModel(BaseModel):

    def __init__(self):
        self.model = LinearRegression()


class PolynormialModel(BaseModel):

    def __init__(self, degree):
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())


class SVMModel(BaseModel):

    def __init__(self, kernel):
        self.model = SVR(kernel=kernel)


class KNNModel(BaseModel):

    def __init__(self, n_neighbors):
        self.model = KNeighborsRegressor(n_neighbors)


class MLPModel(BaseModel):

    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(100,),
                        max_iter=1000,
                        alpha=1e-4,
                        solver='adam',
                        random_state=42,
                        learning_rate_init=0.001)


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

    polynormial_model = PolynormialModel(3)
    avg_r2_poly = polynormial_model.average_r2_cv(dataset_df)
    print(f"avg_r2_poly: {avg_r2_poly}")

    svm_model = SVMModel('rbf')
    avg_r2_svm = svm_model.average_r2_cv(dataset_df)
    print(f"avg_r2_svm: {avg_r2_svm}")

    knn_model = KNNModel(123)
    avg_r2_knn = knn_model.average_r2_cv(dataset_df)
    print(f"avg_r2_knn: {avg_r2_knn}")

    mlp_model = MLPModel()
    avg_r2_mlp = mlp_model.average_r2_cv(dataset_df)
    print(f"avg_r2_mlp: {avg_r2_mlp}")


if __name__ == "__main__":
    #combine(variable_coupling_neurons)
    datasets = ["2022-06-14-13", "2022-07-20-01", "2022-08-02-01",
            "2023-01-23-21"]
    neurons = ["RMED", "RMEV", "RMEL", "RIB", "AIB"]
    num_behavior = 1
    build_heuristics(datasets, neurons, num_behavior)

