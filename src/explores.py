from evaluate import iterate
from query import match_neurons_to_datasets
from tqdm import tqdm
from util import create_train_test_sets, map_neuron_to_heatmap_id, normalize
import cebra
import h5py
import json
import numpy as np
import os
import pandas as pd



def extract_reversals(dataset, neurons):

    processed_h5_path = "/data3/shared/processed_h5_kfc"
    with h5py.File( f"{processed_h5_path}/{dataset}-data.h5", 'r') as f:
        reversal_events = f['behavior']['reversal_events'][:] - 1
        velocity_original = f['behavior']['velocity'][:]
        trace_original = f['gcamp']['trace_array_original'][:]

    velocity_reversals = [
                velocity_original[reversal_events[0][i]:reversal_events[1][i]]
                    for i in range(len(reversal_events[0]))
    ]
    behavior_df = pd.DataFrame(np.concatenate(velocity_reversals),
            columns=['velocity'])

    map_dict = map_neuron_to_heatmap_id(neurons)
    neuron_ids = [map_dict[neuron][dataset] for neuron in neurons]
    trace_reversals = [
                trace_original[reversal_events[0][i]:reversal_events[1][i]]
                    for i in range(len(reversal_events[0]))
    ]
    trace_reversals_subset = [
            trace_reversal[:, neuron_ids]
                for trace_reversal in trace_reversals
    ]

    trace_df = pd.DataFrame(np.concatenate(trace_reversals_subset, axis=0),
                        columns=neurons)
    trace_behavior_df = pd.concat([trace_df, behavior_df], axis=1)
    print(trace_df)
    print(trace_behavior_df)
    return trace_df, behavior_df


def explore_1():

    locomotion_neurons = ['AVB', 'RIB', 'RID', 'RME', 'VB02', 'AVA', 'AVE',
                        'AIB', 'RIM', 'RMED']
    locomotion_datasets = match_neurons_to_datasets(locomotion_neurons)
    print(f'locomotion_datasets: {locomotion_datasets}')
    turning_neurons = ['SAAR', 'SMBR', 'SMDD', 'SIAV', 'SIBL', 'RIVR']
    turning_datasets = match_neurons_to_datasets(turning_neurons)
    print(f'turning_datasets: {turning_datasets}')



def explore_0(ds_name):

    '''Search for best CEBRA model that predicts difference in turning angle
    from a selection of neural traces (and their derivatives)
    '''
    parameter_grid = dict(
        model_architecture="offset10-model",
        min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=[10],
        max_iterations=[10000],
        learning_rate=[0.0001, 0.001],
        output_dimension=[3, 5, 8],
        num_hidden_units=[8, 16, 32],
        batch_size=1024,
        device='cuda:0',
        verbose=True)

    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 0
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 200, 10)]
    num_label = 2
    save_models_dir = f"/home/alicia/data3_personal/cebra_grid_searches/{ds_name.split('.')[0]}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def concatenate_heatstim_datasets(
                neurons,
                datasets,
                normalization,
                dataset_index,
                max_length=1600):

    '''Take in the assembled dataset in .csv format and rearrange the layout of
    neuronal traces and behaviors such that a new animal's data is concatenated
    to the end time point of the previous animal's data

    Args:
        assembled_dataset: name of the assembled data to rearrange
            (e.g. "SMDD-SMDV/SMDD-SMDV_velocity_f0")
    '''

    new_trace_lists = [[] for _ in range(len(neurons))]
    new_behavior_sin_list = []
    new_behavior_cos_list = []

    processed_h5_path = "/data3/shared/processed_h5_kfc"
    map_dict = map_neuron_to_heatmap_id(neurons)

    h5file_path = '/data3/prj_kfc/data/analysis_dict/turning_data.h5'
    turning_data = h5py.File(h5file_path, 'r')

    num_frames_after_heatstim = 200

    for index, dataset in enumerate(datasets):

        t_stim = turning_data[dataset]['stim_time'][()]
        t_afterstim = t_stim + num_frames_after_heatstim
        animal_behavior = \
            turning_data[dataset]['angle_to_stim_loc'][t_stim:t_afterstim]

        new_behavior_sin_list += list(np.sin(animal_behavior.tolist()))
        new_behavior_cos_list += list(np.cos(animal_behavior.tolist()))

        with h5py.File( f"{processed_h5_path}/{dataset}-data.h5", 'r') as f:
            trace_original = f['gcamp']['trace_array_original'][t_stim:t_afterstim]

        for j, neuron in enumerate(neurons):
            heatmap_id = map_dict[neuron][dataset]
            neural_trace = trace_original[:, heatmap_id]
            new_trace_lists[j] += normalize(neural_trace, normalization).tolist()

    if len(neurons) > 2:
        dir_name = '-'.join(neurons[:3]) + '-X'

    new_dir = f"/home/alicia/data3_personal/cebra_data/{dir_name}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_trace_df = pd.DataFrame(
            {neuron: new_trace_lists[i] for i, neuron in enumerate(neurons)}
    )
    new_behavior_df = pd.DataFrame({
                'cos_turndiff': new_behavior_cos_list,
                'sin_turndiff': new_behavior_sin_list
    })

    new_trace_behavior_df = pd.concat([new_trace_df, new_behavior_df], axis=1)
    new_trace_behavior_df.to_csv(
            f'{new_dir}/{dir_name}_turndiff_f{normalization}_{dataset_index}.csv',
            index=False
    )


if __name__ == "__main__":
    explore_1()
