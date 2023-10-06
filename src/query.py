from itertools import chain, combinations
from tqdm import tqdm
import h5py
import json
import numpy as np
import random


file_path = '/data3/prj_kfc/data/analysis_dict/turning_data.h5'
'''
with h5py.File(file_path, 'r') as f:
    heatstim_datasets = list(f.keys())
    behavior = f['2022-12-21-06']['angle_to_stim_loc'][:]
    t_heatstim = f['2022-12-21-06']['stim_time'][()]
    print(behavior.shape)
    print(t_heatstim)
    print(f'heatstim_datasets: {heatstim_datasets}')
'''

def get_neuron_id(neuron, dataset):

    data_path = "/home/alicia/data3_personal/lab_metadata"

    with open(f"{data_path}/unmerged_neuron_to_datasets.json", "r") as f:
        matches_unmerged = json.load(f)
        unmerged_neurons = list(matches_unmerged.keys())
    with open(f"{data_path}/merged_neuron_to_datasets.json", "r") as f:
        matches_merged = json.load(f)
        merged_neurons = list(matches_merged.keys())

    if neuron in unmerged_neurons:
        return matches_unmerged[neuron][dataset]
    elif neuron in merged_neurons:
        return matches_merged[neuron][dataset]


def find_common_neurons_for_datasets(datasets):

    """Find all the neurons (merged and unmerged) that are contained in the
    given datasets and output a dictionary structured as follows:
        {
            neuron_A: {
                        dataset_1: index1,
                        dataset_2: index2,
                        …
            }
            …
        }
    """
    data_path = "/home/alicia/data3_personal/lab_metadata"

    with open(f"{data_path}/dataset_to_unmerged_neurons.json", "r") as f:
        matches_unmerged = json.load(f)
    with open(f"{data_path}/dataset_to_merged_neurons.json", "r") as f:
        matches_merged = json.load(f)

    # find common neurons shared by the datasets

    all_merged_neurons = []
    all_unmerged_neurons = []
    for dataset in datasets:
        all_merged_neurons.append(list(matches_merged[dataset].keys()))
        all_unmerged_neurons.append(list(matches_unmerged[dataset].keys()))

    set_of_merged_neurons = [set(neurons) for neurons in all_merged_neurons]
    common_merged_neurons = list(set.intersection(*set_of_merged_neurons))

    set_of_unmerged_neurons = [set(neurons) for neurons in all_unmerged_neurons]
    common_unmerged_neurons = list(set.intersection(*set_of_unmerged_neurons))

    # format the output dictionary

    output_dict = dict()
    for merged_neuron in common_merged_neurons:
        output_dict[merged_neuron] = dict()

        for dataset in datasets:
            output_dict[merged_neuron][dataset] = \
                matches_merged[dataset][merged_neuron]

    for unmerged_neuron in common_unmerged_neurons:
        output_dict[unmerged_neuron] = dict()

        for dataset in datasets:
            output_dict[unmerged_neuron][dataset] = \
                matches_unmerged[dataset][unmerged_neuron]

    return output_dict


def find_common_datasets_for_neurons(neurons):

    """Find all the datasets that contain given neurons and output a dictionary
    structured as follows:
        {
            dataset_name: {
                neuron_A: indexA,
                neuron_B: indexB,
                …
            }
            …
        }
    """
    data_path = "/home/alicia/data3_personal/lab_metadata"

    with open(f"{data_path}/unmerged_neuron_to_datasets.json", "r") as f:
        matches_unmerged = json.load(f)
        unmerged_neurons = list(matches_unmerged.keys())
    with open(f"{data_path}/merged_neuron_to_datasets.json", "r") as f:
        matches_merged = json.load(f)
        merged_neurons = list(matches_merged.keys())

    # find common datasets shared by the neurons

    all_datasets = []
    for neuron in neurons:
        if neuron in unmerged_neurons:
            all_datasets.append(list(matches_unmerged[neuron].keys()))
        elif neuron in merged_neurons:
            all_datasets.append(list(matches_merged[neuron].keys()))
        else:
            print(f"{neuron} is not found")

    sets_of_datasets = [set(datasets) for datasets in all_datasets]
    common_datasets = list(set.intersection(*sets_of_datasets))

    # format the output dictionary

    output_dict = dict()
    for common_dataset in common_datasets:
        output_dict[common_dataset] = dict()

        for neuron in neurons:
            if neuron in unmerged_neurons:
                neuron_index = matches_unmerged[neuron][common_dataset]
            elif neuron in merged_neurons:
                neuron_index = matches_merged[neuron][common_dataset]
            else:
                continue

            output_dict[common_dataset][neuron] = neuron_index

    return output_dict


def write_json_neuron_to_datasets(neurons_merged):

    # get unique neurons (either merged or unmerged)
    data_path = "/home/alicia/data3_personal/lab_metadata"
    if neurons_merged:
        input_file = "dataset_to_merged_neurons"
        output_file = "merged_neuron_to_datasets"
    else:
        input_file = "dataset_to_unmerged_neurons"
        output_file = "unmerged_neuron_to_datasets"

    with open(f"{data_path}/{input_file}.json", "r") as f:
        matches = json.load(f)
    all_neurons = []
    for info_dict in matches.values():
        all_neurons += list(info_dict.keys())
    unique_neurons = np.unique(all_neurons)

    # search for datasets that contains each neuron
    output_dict = dict()
    for neuron in unique_neurons:
        output_dict[neuron] = dict()

        for dataset, info_dict in matches.items():
            if neuron in info_dict.keys():
                output_dict[neuron][dataset] = info_dict[neuron]

    with open(f'{data_path}/{output_file}.json', 'w') as f:
        json.dump(output_dict, f, indent=4)


def write_json_dataset_to_neurons(neurons_merged):

    """Read from `analysis_dict['dict_match_dict']` and write a new dictionary
    to JSON file as follows:
        * each key is a dataset's name (e.g. '2022-06-14-01')
        * each value is another dictionary that maps neurons (e.g. 'RMDL')
        to their heatmap id in this dataset (following python's indexing
        convention)

    E.g. output:
    {
        "2022-06-14-01": {
            "RMDL": 4,
            "SMDDR": 56,
            ...
        }
        ...
    }
    """
    output_dict = dict()
    data_path = "/home/alicia/data3_personal/lab_metadata"
    with open(f"{data_path}/analysis_dict.json", "r") as f:
        dict_match_dict = json.load(f)['dict_match_dict']
    datasets = list(dict_match_dict.keys())
    for dataset in tqdm(datasets):
        output_dict[dataset] = dict()
        dataset_infos = dict_match_dict[dataset][0]
        for heatmap_id, info_dict in dataset_infos.items():
            if neurons_merged:
                output_dict[dataset][info_dict['neuron_class']] = \
                    int(heatmap_id) - 1
            else:
                output_dict[dataset][info_dict['label']] = int(heatmap_id) - 1

    if neurons_merged:
        file_name = "dataset_to_merged_neurons"
    else:
        file_name = "dataset_to_unmerged_neurons"

    with open(f'{data_path}/{file_name}.json', 'w') as f:
        json.dump(output_dict, f, indent=4)


def write_json_dataset_to_merged_neurons_v0():

    """Read from `analysis_dict` and write a new dictionary to JSON file as
    follows:
        * each key is a dataset's name (e.g. "2022-06-14-01")
        * each value is another dictionary that maps *merged neurons* (i.o.w.,
        both 'RMDL' and 'RMDR' are kept as 'RME') to their heatmap id in this
        dataset (following python indexing convention)

    E.g. output
    {
         "2022-06-14-01": {
            "RME": 54,
            "RME": 94,
            "RID": 4
            ...
        }
        ...
    }
    NOTE: `analysis_dict[matches]` somehow contains the same info as
        `analysis_dict[dict_match_dict]`
    """
    data_path = "/home/alicia/data3_personal/lab_metadata"
    file_path = f'{data_path}/analysis_dict_matches.json'
    with open(file_path, 'r') as f:
        matches = json.load(f)

    all_neurons = list(matches.keys())

    # find common neurons given datasets
    output_dict = dict()
    for neuron in all_neurons:
        datasets_ids = matches[neuron]
        for dataset, heatmap_id in datasets_ids:
            if dataset not in output_dict.keys():
                output_dict[dataset] = {neuron: heatmap_id}
            else:
                output_dict[dataset][neuron] = heatmap_id

    with open(f"{data_path}/dataset_to_merged_neurons.json", 'w') as f:
        json.dump(output_dict, f, indent=4)


def find_heatstim_datasets_for_neuron_subsets(neurons):

    '''Finds the heat-stim datasets that contain each possible subset of
    neurons (i.e., the power set)'''

    # all the possible combinations
    neuron_powerset = powerset(neurons)
    neuron_subsets = random.sample(neuron_powerset, 10)
    for neuron_subset in neuron_subsets:
        print(f'neuron_subset: {neuron_subset}')
        dataset_subset =  find_common_datasets(neuron_subset)
        heatstim_subset = list(set(dataset_subset) & set(heatstim_datasets))
        print(f'heatstim_subset: {heatstim_subset}')


def find_heatstim_datasets_for_neurons(neurons):

    dataset_subset =  find_common_datasets(neurons)
    heatstim_subset = list(set(dataset_subset) & set(heatstim_datasets))
    print(f'heatstim datasets: {heatstim_subset}')
    return heatstim_subset


def powerset(iterable):

    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


if __name__ == "__main__":
    locomotion_neurons = ['AVB', 'RIB', 'RID', 'RME', 'VB02', 'AVA', 'AVE',
                        'AIB', 'RIM', 'RMED']

    datasets = ["2022-06-14-01", "2022-07-26-01", "2023-01-18-01", "2023-01-19-08"]
    #common_ds = find_common_datasets_for_neurons(locomotion_neurons)
    #print(f'common datasets: {common_ds}')
    output_dict = find_common_neurons_for_datasets(datasets)
    print(output_dict)
