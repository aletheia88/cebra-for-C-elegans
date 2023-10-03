from itertools import chain, combinations
from tqdm import tqdm
import h5py
import json
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

def match_neurons_to_datasets(neurons):

    neuron_to_dataset_dict_path = \
                '/home/alicia/data3_personal/analysis_dict_matches.json'
    with open(neuron_to_dataset_dict_path, 'r') as f:
        matches = json.load(f)
    all_datasets = []

    for neuron in neurons:
        all_datasets.append(matches[neuron])

    sets_of_dates = [set(item[0] for item in lst) for lst in all_datasets]
    common_dates = set.intersection(*sets_of_dates)
    return list(common_dates)


def match_datasets_to_neurons(datasets):

    dataset_to_neuron_dict_path = \
                '/home/alicia/data3_personal/dataset_to_neuron_matches.json'
    with open(dataset_to_neuron_dict_path, 'r') as f:
        matches = json.load(f)
    all_neurons = []

    for dataset in datasets:
        all_neurons.append(matches[dataset])

    sets_of_neurons = [set(item[0] for item in lst) for lst in all_neurons]
    common_neurons = set.intersection(*sets_of_neurons)
    print(f'common neurons: {common_neurons}')
    return common_neurons


def write_unmerged_neuron_to_heatmap_id_dict():

    '''Read from `analysis_dict['dict_match_dict']` and write a new dictionary
    to json file as follows:
        * each key is a dataset's name (e.g. '2022-06-14-01')
        * each value is another dictionary that maps neurons (e.g. 'RMDL')
        that are *not merged* to their heatmap id in this dataset

    E.g. output:
    {
        "2022-06-14-01": {
            "RMDL": 4,
            "SMDDR": 56,
            ...
        }
        ...
    }
    '''
    neuron_to_heatmap_id = dict()

    with open('/home/alicia/data3_personal/analysis_dict.json', 'r') as f:
        dict_match_dict = json.load(f)['dict_match_dict']
    datasets = list(dict_match_dict.keys())
    for dataset in tqdm(datasets):
        neuron_to_heatmap_id[dataset] = dict()
        dataset_infos = dict_match_dict[dataset][0]
        for heatmap_id, info_dict in dataset_infos.items():
            neuron_to_heatmap_id[dataset][info_dict['label']] = int(heatmap_id) - 1

    file_path = '/home/alicia/data3_personal'
    with open(f'{file_path}/unmerged_neuron_to_heatmap_id.json', 'w') as f:
        json.dump(neuron_to_heatmap_id, f, indent=4)


def write_dataset_to_neuron_dict():

    '''Read from `analysis_dict` and write a new dictionary to json file that
    contains mapping from dataset name (e.g. '2022-06-14-01') to neurons in the
    dataset and their heatmap ids'''

    neuron_to_dataset_dict_path = \
            '/home/alicia/data3_personal/analysis_dict_matches.json'
    with open(neuron_to_dataset_dict_path, 'r') as f:
        matches = json.load(f)

    all_neurons = list(matches.keys())

    ### find common neurons given datasets
    dataset_to_neuron_dict = dict()
    for neuron in all_neurons:
        datasets_ids = matches[neuron]
        for dataset, heatmap_id in datasets_ids:
            if dataset not in dataset_to_neuron_dict.keys():
                dataset_to_neuron_dict[dataset] = [[neuron, heatmap_id]]
            else:
                dataset_to_neuron_dict[dataset].append([neuron, heatmap_id])

    with open('/home/alicia/data3_personal/dataset_to_neuron_matches.json', 'w') as f:
        json.dump(dataset_to_neuron_dict, f, indent=4)


def find_heatstim_datasets_for_neuron_subsets(neurons):

    '''Finds the heat-stim datasets that contain each possible subset of
    neurons (i.e., the power set)'''

    # all the possible combinations
    neuron_powerset = powerset(neurons)
    neuron_subsets = random.sample(neuron_powerset, 10)
    for neuron_subset in neuron_subsets:
        print(f'neuron_subset: {neuron_subset}')
        dataset_subset =  match_neurons_to_datasets(neuron_subset)
        heatstim_subset = list(set(dataset_subset) & set(heatstim_datasets))
        print(f'heatstim_subset: {heatstim_subset}')


def find_heatstim_datasets_for_neurons(neurons):

    dataset_subset =  match_neurons_to_datasets(neurons)
    heatstim_subset = list(set(dataset_subset) & set(heatstim_datasets))
    print(f'heatstim datasets: {heatstim_subset}')
    return heatstim_subset


def powerset(iterable):

    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


if __name__ == "__main__":
