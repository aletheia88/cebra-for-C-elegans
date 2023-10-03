from collections import Counter
from scipy.interpolate import CubicSpline
from scipy.stats import zscore
import copy
import csv
import h5py
import json
import numbers
import numpy as np
import os
import pandas as pd
import random


DS_PATH = "/home/alicia/data_personal/cebra_data"


def concatenate(neurons, behavior, normalization, dataset_index,
        max_length=1600):

    '''Concatenate neural traces and behavior to the end time point of the
    previous animal's data; the final time series is from multiple animals

    Args:
        neurons: a list of neurons to extract traces from
        behavior: the animal behavior
        normalization: factor to divide the raw neural traces by
        dataset_index: index of the new dataset; typically given as the number
            of animals in the final concatenated dataset
    '''

    new_trace_lists = [[] for _ in range(len(neurons))]
    new_behavior_list = []

    processed_h5_path = "/data1/shared/processed_h5_kfc"
    map_dict = map_neuron_to_heatmap_id(neurons)

    all_datasets = []
    for datasets_ids in list(map_dict.values()):
        all_datasets += list(datasets_ids.keys())

    shared_datasets = get_items_occurring_k_times(all_datasets, len(neurons))
    print(f'shared_datasets: {shared_datasets}')

    for index, dataset in enumerate(shared_datasets):
        with h5py.File( f"{processed_h5_path}/{dataset}-data.h5", 'r') as f:
            trace_original = f['gcamp']['trace_array_original'][:max_length]
            animal_behavior = f['behavior'][behavior][:max_length]

        new_behavior_list += animal_behavior.tolist()
        for j, neuron in enumerate(neurons):
            heatmap_id = map_dict[neuron][dataset]
            neural_trace = trace_original[:, heatmap_id]
            new_trace_lists[j] += normalize(neural_trace, normalization).tolist()

    dir_name = '-'.join(neurons)
    new_dir = f"/home/alicia/data_personal/cebra_data/{dir_name}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_trace_df = pd.DataFrame(
            {neuron: new_trace_lists[i] for i, neuron in enumerate(neurons)}
    )
    new_behavior_df = pd.DataFrame({behavior: new_behavior_list})
    new_trace_behavior_df = pd.concat([new_trace_df, new_behavior_df], axis=1)
    new_trace_behavior_df.to_csv(
            f'{new_dir}/{dir_name}_{behavior}_f{normalization}_{dataset_index}.csv',
            index=False
    )


def subset(ds_path, ds_names, neurons, new_ds_index, behaviors=None):

    '''Obtain a sub-selection of datasets that contains the neuronal traces of
    the specified neurons, as well as behaviors if any

    Args:
        ds_path: dataset name and directory under DS_PATH to the full dataset
            e.g. 'SMDD-SMDV/SMDD-SMDV_velocity_f10.csv'
        ds_names: a list of dataset names that the subset should contain
        neurons: a list of neuron types, e.g. ['RIB', 'AVE']
        behaviors: a list of behaviors, e.g. ['velocity']; default=None
    '''

    dir_name, full_ds_name = ds_path.split('/')
    df = pd.read_csv(f'{DS_PATH}/{ds_path}')
    neural_columns = [f"{neuron}_{ds_name}"
            for neuron, ds_name in zip(neurons*len(ds_names),
                ds_names*len(neurons))]
    if behaviors is not None:
        behavior_columns = [f"{behavior}_{ds_name}"
            for behavior, ds_name in zip(behaviors*len(ds_names),
                ds_names*len(behaviors))]
    else:
        behavior_columns = []

    return df[neural_columns + behavior_columns].to_csv(
            f"{DS_PATH}/{dir_name}/{full_ds_name.split('.')[0]}_{new_ds_index}.csv",
            index=False
        )


def assemble(neurons, behavior, normalization, max_length=1600):

    '''Assemble the neural traces of the given neurons together with their
    behaviors across time and write to a csv file

    Args:
        neurons: a list of neurons to obtain neural traces
        behavior: animal behavior to be associated with the given neurons; e.g.
            velocity, pumping, head curvature
        normalization: the choice of normalization; current options are
            - 10 := normalizing raw traces by dividing the 10th percentile
              fluorescence
            - 20 := normalizing raw traces by dividing the 20th percentile
              fluorescence
            - "z" := replacing each neural activity measurement with its
              z-score
            - "cube" := the cube root of the traces normalized by dividing the
              10th percentile
    '''

    new_trace_df = pd.DataFrame()
    new_behavior_df = pd.DataFrame()
    processed_h5_path = "/data1/shared/processed_h5_kfc"
    map_dict = map_neuron_to_heatmap_id(neurons)

    all_datasets = []
    for datasets_ids in list(map_dict.values()):
        all_datasets += list(datasets_ids.keys())

    shared_datasets = get_items_occurring_k_times(all_datasets, len(neurons))

    for index, dataset in enumerate(shared_datasets):
        with h5py.File(f"{processed_h5_path}/{dataset}-data.h5", 'r') as f:
            trace_original = f['gcamp']['trace_array_original'][:max_length]
            animal_behavior = f['behavior'][behavior][:max_length]

        new_behavior_df[f"{behavior}_{dataset}"] = animal_behavior

        for neuron in neurons:
            heatmap_id = map_dict[neuron][dataset]
            neural_trace = trace_original[:, heatmap_id]
            normalized_neural_trace = normalize(neural_trace, normalization)
            new_trace_df[f"{neuron}_{dataset}"] = normalized_neural_trace

    dir_name = '-'.join(neurons)
    new_dir = f"/home/alicia/data_personal/cebra_data/{dir_name}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    '''
    new_trace_df.to_csv(f'{new_dir}/{dir_name}_f{normalization}.csv',
            index=False)
    new_behavior_df.to_csv(f'{new_dir}/{dir_name}_{behavior}.csv', index=False)
    '''
    new_trace_behavior_df = pd.concat([new_trace_df, new_behavior_df], axis=1)
    new_trace_behavior_df.to_csv(
                f'{new_dir}/{dir_name}_{behavior}_f{normalization}.csv',
                index=False)


def normalize(neural_trace, normalization):

    '''Normalize the raw neuronal trace

    Args:
        neural_trace: an array of neural activities at each time point
        normalization: the choice of normalization; current options are
            - 10 := normalizing raw traces by dividing the 10th percentile
              fluorescence
            - 20 := normalizing raw traces by dividing the 20th percentile
              fluorescence
            - "z" := replacing each neural activity measurement with its
              z-score
            - "cube" := the cube root of the traces normalized by dividing the
              10th percentile
    '''

    if isinstance(normalization, numbers.Number):
        factor = np.percentile(neural_trace, normalization)
        normalized_neural_trace = neural_trace / factor
    elif normalization == 'z':
        normalized_neural_trace = zscore(neural_trace)
    elif normalization == 'cube':
        factor = np.percentile(neural_trace, 10)
        normalized_neural_trace = np.cbrt(neural_trace / factor)

    return normalized_neural_trace


def map_neuron_to_heatmap_id(neurons):

    '''Takes a list of neurons and find the datasets that contain them; match
    each neuron to its heatmap id in each of these datasets in the following
    format:
    {
        'SMDV':
            {
                '2022-06-14-01': 47,
                '2022-06-14-01', 96,
                ...
            }
    }
    Args:
        neurons: a list of neuron names, e.g. SMDD, RIB, AVE
    '''

    home_path = '/home/alicia/data_personal'
    with open(f'{home_path}/analysis_dict_matches.json', 'r') as f:
        matches = json.load(f)

    map_dict = dict()
    for neuron in neurons:
        map_dict[neuron] = dict()
        datasets_ids = matches[neuron]
        for dataset, heatmap_id in datasets_ids:
            map_dict[neuron][dataset] = heatmap_id - 1
    return map_dict


def get_items_occurring_k_times(lst, k):
    count = Counter(lst)
    return [item for item, freq in count.items() if freq == k]


def create_train_test_sets(ds_name,
                           noise_ds_name,
                           train_ratio,
                           num_train_test_splits,
                           num_augmentations,
                           noise_multiplier,
                           num_label):

    df = pd.read_csv(f"{DS_PATH}/{ds_name}")

    # sample consecutive indices
    num_samples = df.shape[0]
    num_train_samples = int(num_samples * train_ratio)
    num_test_samples = num_samples - num_train_samples

    # possible start indices for train data
    train_start_indices = random.sample(range(0, num_samples - num_train_samples + 1),
                            num_train_test_splits)
    # create train and test dataframes from original data (w/o noise)
    train_base_dfs = [
                df.iloc[s:s+num_train_samples] for s in train_start_indices
    ]
    test_dfs = [
                df.iloc[
                    list(set(range(num_samples)) - set(range(s, s + num_train_samples)))
                ] for s in train_start_indices
    ]
    # separate test neural data and test behavior labels from test dataframes
    neural_tests = [df.iloc[:, :-num_label].values for df in test_dfs]
    label_tests = [df.iloc[:, -num_label:].values for df in test_dfs]
    neural_trains = [df.iloc[:, :-num_label].values for df in train_base_dfs]
    label_trains = [df.iloc[:, -num_label:].values for df in train_base_dfs]

    augmented_neural_tests = []
    augmented_neural_trains = []
    augmented_label_trains = []

    for i in range(num_train_test_splits):
        # augment test data by adding its derivatives
        augmented_neural_test, _ = augment(neural_tests[i],
                                      label_tests[i],
                                      noise_ds_name,
                                      noise_multiplier,
                                      num_label,
                                      num_noise_augment=0)
        augmented_neural_tests.append(augmented_neural_test)

        # augment train data by adding noise and derivatives
        augmented_neural_train, augmented_label_train = augment(
                                      neural_trains[i],
                                      label_trains[i],
                                      noise_ds_name,
                                      noise_multiplier,
                                      num_label,
                                      num_noise_augment=num_augmentations)
        augmented_neural_trains.append(augmented_neural_train)
        augmented_label_trains.append(augmented_label_train)

    return {"neural_tests": augmented_neural_tests,
            "label_tests": label_tests,
            "neural_trains": augmented_neural_trains,
            "label_trains": augmented_label_trains}


def augment(augmented_neural,
            behavior_data,
            noise_ds_name,
            noise_multiplier,
            num_label,
            num_noise_augment):

    if num_noise_augment == 0:
        derivatives = get_matrix_derivative(augmented_neural)
        augmented_data = np.hstack((augmented_neural, derivatives))
        concat_augmented_behaviors = copy.deepcopy(behavior_data)

    else:
        original = copy.deepcopy(augmented_neural[:])
        derivatives = get_matrix_derivative(original)
        GFP_std_dev = np.std(pd.read_csv(f"{DS_PATH}/{noise_ds_name}").values)

        for i in range(num_noise_augment):
            new_neural_data = original + noise_multiplier * np.random.normal(0,
                                    GFP_std_dev, original.shape)
            augmented_neural = np.vstack((augmented_neural, new_neural_data))
            derivatives = np.vstack((derivatives,
                        get_matrix_derivative(new_neural_data)))

        augmented_data = np.hstack((augmented_neural, derivatives))
        augmented_behaviors = np.tile(behavior_data, num_noise_augment + 1)
        ### TODO: delete
        print(augmented_behaviors.shape)
        concat_augmented_behaviors = np.concatenate([
                    augmented_behaviors[:, i].ravel()
                    for i in range(augmented_behaviors.shape[1])
                ]
            )

    return augmented_data, concat_augmented_behaviors


def get_matrix_derivative(data):

    matrix_derivative = copy.deepcopy(data)
    num_columns = data.shape[1]
    for j in range(num_columns):
        matrix_derivative[:, j] = get_vector_derivative(matrix_derivative[:, j])
    return matrix_derivative


def derivative(x, y, nu=1):
    spline = CubicSpline(x, y)
    return spline.derivative(nu)(x)


def get_vector_derivative(y, nu=1):
    x = np.arange(1, np.size(y) + 1)
    spline = CubicSpline(x, y)
    return spline.derivative(nu)(x)


if __name__ == "__main__":
    neurons = ['RIB', 'AVE']
    behavior = 'velocity'
    normalization = 'cube'
    dataset_index = 54
    ds_path = 'SMDD-SMDV/SMDD-SMDV_velocity_f10.csv'
    ds_names = ['2022-06-14-01']

    concatenate(neurons, behavior, normalization, dataset_index)
    #assemble(neurons, behavior, normalization)
    #subset(ds_path, ds_names, neurons, 1, behaviors=[behavior])
