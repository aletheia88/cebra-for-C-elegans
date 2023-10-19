from evaluate import iterate
from fit import fit_linear_model
from query import get_neuron_id
from tqdm import tqdm
from utils import create_train_test_sets, normalize
import cebra
import copy
import h5py
import json
import numpy as np
import os
import pandas as pd


def benchmark(dataset_name):

    average_r2 = fit_linear_model(dataset_name)
    print(f"average r^2: {average_r2}")


def run():

    """
    run locomotion neuron experiments
        - experiment: RMED-RMEL-RIB_reveral-velocity_f10_7
            * neurons: RMED RMEL RIB AVE
            * number of animals: 7
        - experiment: RMED-RMEV-RMEL_reveral-velocity_f10_4
            * neurons: RMED RMEV RMEL RIB AIB
            * number of animals: 4
    """
    dataset_1_name = \
        "RMED-RMEV-RMEL/RMED-RMEV-RMEL_reversal-velocity_f10_4_linearized.csv"
    dataset_2_name = \
        "RMED-RMEL-RIB/RMED-RMEL-RIB_reversal-velocity_f10_7_linearized.csv"

    """
    run tuning neuron experiments
        - experiment: SAADR-SAADL-SMDDL_reversal-velocity_f10_3
            * neurons: SAADR SAADL SMDDL SMDDR SMDVL RIVL RIVR
            * number of animals: 3
    """
    dataset_3_name = \
        "SAADR-SAADL-SMDDL/SAADR-SAADL-SMDDL_reversal-velocity_f10_3_linearized.csv"

    """
    run inconsistent-locomotion-neuron experiments
        - experiment: AUAL-AIML-AIYL_reversal-velocity_f10_3
            * neurons: AUAL AIML AIYL URYDL URYVL BAGL ASGL
            * number of animals: 4
    """
    dataset_4_name = \
        "AUAL-AIML-AIYL/AUAL-AIML-AIYL_reversal-velocity_f10_4_linearized.csv"
    """
    run variable-coupling neuron experiments
        - experiment: RIR-RIH-URBL_reversal-velocity_f10_4_linearized
            * neurons: 'RIR', 'RIH', 'URBL', 'RMDDL', 'IL1R', 'IL1L'
            * number of animals: 4
    """
    dataset_5_name = \
        "RIR-RIH-URBL/RIR-RIH-URBL_reversal-velocity_f10_4_linearized.csv"
    explore(dataset_5_name)


def explore(dataset_name):

    data_path = "/home/alicia/data3_personal/cebra_data"
    parameter_grid = dict(
        model_architecture="offset10-model",
        min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=10,
        max_iterations=1,
        learning_rate=[0.0001, 0.001],
        output_dimension=[3, 5, 8],
        num_hidden_units=[8, 16, 32],
        batch_size=None,
        device='cuda:1',
        #device="cuda_if_available",
        verbose=True)

    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 0
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 200, 10)]
    num_label = 1

    experiment_dir, experiment_name = dataset_name.split('.')[0].split('/')

    output_path = "/home/alicia/data3_personal/cebra_grid_searches"

    if not os.path.exists(f"{output_path}/{experiment_dir}"):
        os.mkdir(f"{output_path}/{experiment_dir}")

    save_models_dir = f"{output_path}/{experiment_dir}/{experiment_name}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, dataset_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def concatenate_reversal_datasets(datasets,
                                  neurons,
                                  normalization,
                                  linearize,
                                  export_csv):

    """Concatenate all the given animal datasets by appending the next animal's
    neural activities and behaviors to the ones of the previous animal

    Args:
        datasets: a list of dataset names; e.g. ['2023-01-10-07',
                '2023-01-10-14', '2023-01-16-08', '2023-01-23-15']
        neurons: a list of neuron names
    """
    new_behavior_list = []
    new_trace_list = []
    for dataset in datasets:
        trace_df, behavior_df = extract_reversal_timepoints(
                    dataset,
                    neurons,
                    normalization)
                    #linearize)
        new_behavior_list.append(behavior_df)
        new_trace_list.append(trace_df)

    dir_name = '-'.join(neurons[:4])
    new_dir = f"/home/alicia/data3_personal/cebra_data/{dir_name}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    new_trace_df = pd.concat(new_trace_list, ignore_index=True)
    new_behavior_df = pd.concat(new_behavior_list, ignore_index=True)
    new_trace_behavior_df = pd.concat([new_trace_df, new_behavior_df], axis=1)

    if export_csv:
        # index dataset by the number of animals concatenated
        dataset_index = len(datasets)
        if linearize:
            new_file_name = \
                f"{dir_name}_reversal-timepoints_f{normalization}_{dataset_index}_linearized"
        else:
            new_file_name = \
                f"{dir_name}_reversal-timepoints_f{normalization}_{dataset_index}"
        new_trace_behavior_df.to_csv(
                f"{new_dir}/{new_file_name}.csv",
                index=False
        )
        print(f"file exported at {new_dir}/{new_file_name}.csv")
    else:
        return new_trace_behavior_df


def extract_reversal_timepoints(dataset, neurons, normalization):

    processed_h5_path = "/data3/shared/processed_h5_kfc"
    with h5py.File(f"{processed_h5_path}/{dataset}-data.h5", 'r') as f:
        reversal_events = f['behavior']['reversal_events'][:] - 1
        trace_original = f['gcamp']['trace_array_original'][:]

    # normalize neural traces of interest

    neuron_ids = [get_neuron_id(neuron, dataset) for neuron in neurons]
    trace_normalized = copy.deepcopy(trace_original)
    for neuron_id in neuron_ids:
        trace_normalized[:, neuron_id] = normalize(
                    trace_original[:, neuron_id],
                    normalization
        )
    trace_reversals = [
                trace_normalized[reversal_events[0][i]:reversal_events[1][i]+1]
                    for i in range(len(reversal_events[0]))
    ]
    trace_reversals_subset = [
            trace_reversal[:, neuron_ids]
                for trace_reversal in trace_reversals
    ]
    trace_df = pd.DataFrame(
            np.concatenate(
                    trace_reversals_subset,
                    axis=0),
            columns=neurons)

    reversal_timepoints = []
    for ith_event in range(len(reversal_events[0])):

        reversal_start = reversal_events[0][ith_event]
        reversal_end = reversal_events[1][ith_event]
        reversal_length = reversal_end - reversal_start
        reversal_timepoints += list(range(reversal_length, -1, -1))

    behavior_df = pd.DataFrame(
            np.array(reversal_timepoints).astype(float),
            columns=['timepoints_from_reversal_ends'])

    return trace_df, behavior_df


def extract_reversals(dataset, neurons, normalization, linearize):

    """Extract segments from the given dataset's neural activities and animal
    behaviors where reversal happens and concatenate the extracted segments to
    create a new dataframe

    Args:
        dataset: name of the dataset; e.g. '2023-01-23-15'
        neurons: a list of neuron to extract neural traces from
    """
    processed_h5_path = "/data3/shared/processed_h5_kfc"
    with h5py.File( f"{processed_h5_path}/{dataset}-data.h5", 'r') as f:
        reversal_events = f['behavior']['reversal_events'][:] - 1
        velocity_original = f['behavior']['velocity'][:]
        trace_original = f['gcamp']['trace_array_original'][:]

    # normalize neural traces of interest

    neuron_ids = [get_neuron_id(neuron, dataset) for neuron in neurons]
    trace_normalized = copy.deepcopy(trace_original)
    for neuron_id in neuron_ids:
        trace_normalized[:, neuron_id] = normalize(
                    trace_original[:, neuron_id],
                    normalization
        )

    #print(f"reversal_events: {reversal_events}")
    if linearize:
        velocity_reversals = [
                    linearize_trace(velocity_original[reversal_events[0][i]:reversal_events[1][i]])
                        for i in range(len(reversal_events[0]))
        ]
    else:
        velocity_reversals = [
                    velocity_original[reversal_events[0][i]:reversal_events[1][i]]
                        for i in range(len(reversal_events[0]))
        ]
    behavior_df = pd.DataFrame(np.concatenate(velocity_reversals),
            columns=['velocity'])

    trace_reversals = [
                trace_normalized[reversal_events[0][i]:reversal_events[1][i]]
                    for i in range(len(reversal_events[0]))
    ]
    trace_reversals_subset = [
            trace_reversal[:, neuron_ids]
                for trace_reversal in trace_reversals
    ]
    trace_df = pd.DataFrame(
            np.concatenate(
                    trace_reversals_subset,
                    axis=0),
            columns=neurons)
    trace_behavior_df = pd.concat([trace_df, behavior_df],
                axis=1,
                ignore_index=True)

    return trace_df, behavior_df


def linearize_trace(trace):

    return np.linspace(trace[0], trace[-1], len(trace))


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

    neurons = ["IL1DL", "IL1DR", "IL1VL", "IL1VR", "IL1L", "IL1R"]
    datasets = ['2022-07-15-12', '2023-01-05-01', '2023-01-06-01',
            '2023-01-16-15',
            '2023-01-16-22', '2023-01-19-01', '2023-01-23-08']
    normalization = 10
    linearize = False
    export_csv = True
    """concatenate_reversal_datasets(datasets,
                                  neurons,
                                  normalization,
                                  linearize,
                                  export_csv)"""
    dataset_name = \
            "IL1DL-IL1DR-IL1VL-IL1VR/IL1DL-IL1DR-IL1VL-IL1VR_reversal-timepoints_f10_7.csv"
    explore(dataset_name)
