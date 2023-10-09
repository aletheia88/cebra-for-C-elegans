from decode import linear_decode, knn_decode
from evaluate import iterate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from utils import create_train_test_sets
import cebra
import h5py
import json
import os
import pandas as pd
import random
import torch


KNN_NEIGHBORS = [2*n+1 for n in range(1, 200, 10)]


def fit_multi_session_cebra(
            neurons,
            noise_ds_name='2022-01-07-03/2022-01-07-03_F20.csv',
            train_ratio=0.8,
            num_train_test_splits=5,
            num_augments=10,
            noise_multiplier=1,
            num_neighbors=KNN_NEIGHBORS):

    '''Run a grid search of single-session CEBRA models that train embeddings
    on a assembled dataset of many animals--such dataset typically only
    contains a few neurons'''

    ## TODO: rewrite this function
    # create train-test splits
    ds_path = f'/home/alicia/data_personal/cebra_data/{neurons}'
    neural_data = pd.read_csv(f'{ds_path}/{neurons}_f10_0.csv').to_numpy()
    behavior_data = pd.read_csv(f'{ds_path}/{neurons}_velocity_0.csv').to_numpy()
    neural_data_series = preprocess(neural_data)
    #behavior_data_series = preprocess(behavior_data)
    behavior_data_series = [behavior_data[:, i] for i in range(behavior_data.shape[1])]
    # define grid search parameters
    params_grid = dict(
            min_temperature=[0.01, 0.1, 1],
            temperature_mode = "auto",
            time_offsets=[10],
            batch_size=1024,
            max_iterations=[10000],
            learning_rate=[0.001, 0.0001],
            output_dimension=[3, 5, 8],
            num_hidden_units=[16],
            device='cuda:1',
            verbose=True)

    save_models_dir = f'/home/alicia/data_personal/cebra_outputs/{neurons}'
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)
    #model.fit(neural_data_series, behavior_data_series)
    #model_name = f'min_temperature_0.01_num_hidden_units_16_output_dimension_3_CEBRA-behavior'
    #model.save(f'{save_models_dir}/{model_name}.pt')

    datasets = {"CEBRA-behavior": (neural_data_series, behavior_data_series)}
    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets,
            params=params_grid,
            models_dir=save_models_dir)


def preprocess(data, num_per_session=2):

    '''Remove NaN values and break data into a list of time series

    Args:
        data: numpy array of shape (num time points, num neurons)
        num_per_session: number of neurons to be included per session
    '''
    data_series = [data[:, i:i+num_per_session]
            for i in range(0, data.shape[1], num_per_session)]
    '''
    for series in data_series:
        print(f"series:{series.shape} {series}")
        filtered_series = series[~np.isnan(series).any(axis=1)]
        print(f'filtered series:{filtered_series.shape} {filtered_series}')
    '''
    return [series[~np.isnan(series).any(axis=1)] for series in data_series]


def fit_single_session_cebra(
            ds_name,
            noise_ds_name='2022-01-07-03/2022-01-07-03_F20.csv',
            train_ratio=0.8,
            num_train_test_splits=5,
            num_augments=10,
            noise_multiplier=1,
            num_neighbors=KNN_NEIGHBORS,
            num_label=1):

    '''Run a grid search of single-session CEBRA models that train embeddings
    on a chosen dataset'''

    save_models_dir = \
        f"/home/alicia/data_personal/cebra_grid_searches/{ds_name.split('.c')[0]}"
    if not os.path.exists(save_models_dir):
        os.makedirs(save_models_dir)

    parameter_grid = dict(
            min_temperature=[0.01, 0.1, 1],
            temperature_mode = "auto",
            time_offsets=[10],
            max_iterations=[20000],
            learning_rate=[0.0001],
            output_dimension=[3],
            num_hidden_units=[16, 32],
            device='cuda:0',
            verbose=True)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio, num_train_test_splits,
            num_augments, noise_multiplier, num_neighbors, num_label, save_models_dir)


def fit_linear_model(ds_name,
                     train_ratio=0.8,
                     num_train_test_splits=5,
                     num_augmentations=0,
                     noise_multiplier=1,
                     num_label=1):

    '''Train a linear decoder that maps neuronal activities to behavior(s);
    then report the test-time performance in terms of the R^2 score

    Arg:
        ds_name: name of the dataset from which to fit a linear regression
        (e.g. "SMDD-SMDV/SMDD-SMDV_velocity_f10_8.csv")
    '''

    model = LinearRegression()
    aggregate_r2_score = 0
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_test_sets = create_train_test_sets(ds_name, noise_ds_name,
            train_ratio, num_train_test_splits, num_augmentations,
            noise_multiplier, num_label)
    neural_trains = train_test_sets["neural_trains"]
    label_trains = train_test_sets["label_trains"]
    neural_tests = train_test_sets["neural_tests"]
    label_tests = train_test_sets["label_tests"]
    print(neural_trains[0].shape)
    for i in range(num_train_test_splits):

        model.fit(neural_trains[i], label_trains[i])
        predicted_label = model.predict(neural_tests[i])
        r2 = r2_score(label_tests[i], predicted_label)
        print(f'r2: {r2}')
        aggregate_r2_score += r2

    return aggregate_r2_score / num_train_test_splits
