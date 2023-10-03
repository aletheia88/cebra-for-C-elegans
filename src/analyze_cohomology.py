"""Utilities for applying cohomology analysis on CEBRA embeddings, including
computing Betti numbers and creating persistence diagrams"""

from persim import plot_diagrams
from util import get_matrix_derivative
import cebra
import json
import numpy as np
import os
import pandas as pd
import random
import ripser
import time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def plot_lifespan(topology_dgms, shuffled_max_lifespan, ax, label_vis, maxdim):

    plot_diagrams(
        topology_dgms,
        ax=ax,
        legend=True,
    )

    ax.plot(
        [
            -0.5,
            2,
        ],
        [-0.5 + shuffled_max_lifespan[0], 2 + shuffled_max_lifespan[0]],
        color="C0",
        linewidth=3,
        alpha=0.5,

    )
    ax.plot(
        [
            -0.5,
            2,
        ],
        [-0.5 + shuffled_max_lifespan[1], 2 + shuffled_max_lifespan[1]],
        color="orange",
        linewidth=3,
        alpha=0.5,

    )
    if maxdim == 2:
        ax.plot(
            [-0.50, 2],
            [-0.5 + shuffled_max_lifespan[2], 2 + shuffled_max_lifespan[2]],
            color="green",
            linewidth=3,
            alpha=0.5,
        )
    ax.set_xlabel("Birth", fontsize=15)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([0, 1, 2])
    ax.tick_params(labelsize=13)
    ax.set_aspect('equal')
    if label_vis:
        ax.set_ylabel("Death", fontsize=15)
    else:
        ax.set_ylabel("")


def compute_robust_betti_number(ripser_output,
                                ds_name,
                                model_parameters,
                                num_shuffled_models,
                                maxdim):

    shuffled_max_lifespan = compute_shuffled_max_lifespan(
                              ds_name,
                              model_parameters,
                              num_shuffled_models,
                              maxdim)

    print(f'shuffled_max_lifespan: {shuffled_max_lifespan}')
    return get_betti_number(ripser_output, shuffled_max_lifespan)


def compute_shuffled_max_lifespan(ds_name,
                              model_parameters,
                              num_shuffled_models,
                              maxdim,
                              add_derivatives=True):

    # step 0: create one a dataset with shuffled labels
    # step 1: fit a CEBRA model
    # step 2: transform the dataset to obtain an embedding
    # step 3: applying cohomology analysis on shuffled-CEBRA embedding
        # -> shuffled_ripser_output
    # step 4: compute its maximum lifespan
        # -> shuffled_max_lifespan

    ds_path = '/home/alicia/data_personal/cebra_data'
    df = pd.read_csv(f'{ds_path}/{ds_name}')
    neural_data = df.iloc[:, :-1].values
    behaviors = df.iloc[:, -1].values
    if add_derivatives:
        derivatives = get_matrix_derivative(neural_data)
        neural_data = np.hstack((neural_data, derivatives))

    model = cebra.CEBRA(
                 model_architecture = "offset10-model",
                 batch_size = model_parameters["batch_size"],
                 temperature_mode = "auto",
                 min_temperature = model_parameters["min_temperature"],
                 max_iterations = model_parameters["max_iterations"],
                 learning_rate = model_parameters["learning_rate"],
                 num_hidden_units = model_parameters["num_hidden_units"],
                 output_dimension = model_parameters["output_dimension"],
                 device = model_parameters["device"],
                 verbose=True
            )

    shuffled_lifespans = []
    for model_index in range(num_shuffled_models):

        random.shuffle(behaviors)
        model.fit(neural_data, behaviors)
        embedding = model.transform(neural_data)

        '''
        # save model
        model_subdir, experiment_name = ds_name.split('.')[0].split('/')
        model_path = f"/home/alicia/data_personal/shuffled_cebra_models/{model_subdir}"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model.save(f"{model_path}/{experiment_name}_{model_index}.pt")
        '''
        random_idx = np.random.permutation(np.arange(len(embedding)))[:1000]
        # apply cohomology analysis on embedding
        shuffled_ripser_output = io_ripser_output(
                        ds_name.split('.')[0],
                        0,
                        0,
                        maxdim,
                        add_derivatives,
                        embedding=embedding[random_idx])
        shuffled_lifespan, lifespan_dict = get_max_lifespan(
                        shuffled_ripser_output,
                        maxdim)
        shuffled_lifespans.append(shuffled_lifespan)

    shuffled_max_lifespan = [
                max(shuffled_lifespans, key=lambda x: x[0])[0],
                max(shuffled_lifespans, key=lambda x: x[1])[1],
                max(shuffled_lifespans, key=lambda x: x[2])[2]
    ]
    print(f'shuffled_max_lifespan: {shuffled_max_lifespan}')
    print(len(shuffled_lifespans))
    print(shuffled_lifespans[0])
    print(len(shuffled_lifespans[0]))
    print(shuffled_lifespans)

    return shuffled_max_lifespan


def io_ripser_output(experiment_name,
                     ds_name,
                     model_name,
                     maxdim,
                     add_derivatives,
                     embedding=None,
                     write_to_json=False):

    """Apply cohomology analysis and write the results to .json file

    Args:
        experiment_name: name of the experiment; e.g.
            'SMDD-SMDV/SMDD-SMDV_velocity_f10_25'
        ds_name: name of the dataset used in the experiment; e.g.
            'SMDD-SMDV/SMDD-SMDV_velocity_f10_25.csv'
        model_name: name of the CEBRA model fitted on the dataset; e.g.
            'min_temperature_1_num_hidden_units_32_CEBRA-behavior.pt'
        maxdim: maximum dimension to compute lifespan up to; 0 corresponds to
            H0, 1 to H1, and 2 to H2, etc.
    """

    if embedding is None:
        model_search_path = '/home/alicia/data_personal/cebra_grid_searches'
        model = cebra.CEBRA.load(f'{model_search_path}/{experiment_name}/{model_name}')
        ds_path = '/home/alicia/data_personal/cebra_data'
        df = pd.read_csv(f'{ds_path}/{ds_name}')
        neural_data = df.iloc[:, :-1].values

        if add_derivatives:
            derivatives = get_matrix_derivative(neural_data)
            neural_data = np.hstack((neural_data, derivatives))

        # generate embedding
        embedding = model.transform(neural_data)

    # apply cohomology analysis
    np.random.seed(111)
    random_idx = np.random.permutation(np.arange(len(embedding)))[:1000]
    ripser_output = ripser.ripser(embedding[random_idx],
                                  maxdim=maxdim,
                                  coeff=47)

    if write_to_json:
        file_name = f"ripser_output_{experiment_name.split('/')[1]}_H{maxdim}.json"
        file_path = "/home/alicia/data_personal/ripser_outputs"
        with open(f'{file_path}/{file_name}', 'w') as f:
            json.dump(ripser_output, f, indent=4, default=numpy_array_handler)

    return ripser_output


def get_max_lifespan(ripser_output, maxdim):

    lifespan_dict = {i: [] for i in range(maxdim + 1)}
    for dim in range(maxdim + 1):
        lifespan = read_lifespan(ripser_output, dim)
        lifespan_dict[dim].extend(lifespan)

    return [max(lifespan_dict[i])
            if lifespan_dict[i]
            else np.array([np.negative(np.inf)])
            for i in range(maxdim+1)], lifespan_dict


def read_lifespan(ripser_output, dim):

    persistence_diagram = np.array(ripser_output['dgms'][dim])
    return persistence_diagram[:, 1] - persistence_diagram[:, 0]


def get_betti_number(ripser_output, shuffled_max_lifespan):

    bettis = []
    dgms = ripser_output['dgms']

    for dim in range(len(dgms)):
        persistence_diagram = np.array(dgms[dim])
        lifespans = persistence_diagram[:, 1] - persistence_diagram[:, 0]
        betti_d = sum(lifespans > shuffled_max_lifespan[dim] * 1.1)
        print(f"betti_d = {betti_d}")
        bettis.append(betti_d)

    return bettis


def numpy_array_handler(obj):

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def prep():

    #experiment_name = "SMDD-SMDV/SMDD-SMDV_velocity_f10_25"
    #ds_name = "SMDD-SMDV/SMDD-SMDV_velocity_f10_25.csv"
    experiment_name = "RIB-AVE/RIB-AVE_velocity_f10_54"
    ds_name = "RIB-AVE/RIB-AVE_velocity_f10_54.csv"
    model_name = "_CEBRA-behavior.pt"
    model_parameters = {
            "model_architecture": "offset10-model",
            "min_temperature": 0.1,
            "temperature_mode": "auto",
            "time_offsets": [10],
            "max_iterations": [50000],
            "learning_rate": [0.001],
            "output_dimension": [3],
            "num_hidden_units": 16,
            "batch_size": 1024,
            "device": 'cuda:0',
            "verbose": True
    }
    maxdim = 2
    ripser_start = time.time()
    io_ripser_output(experiment_name,
                     ds_name,
                     model_name,
                     maxdim)
    ripser_end = time.time()
    print(f'time taken: {ripser_end - ripser_start}')


def run():
    experiment_name = 'SMDD-SMDV/SMDD-SMDV_velocity_f10_25'
    ds_name = 'SMDD-SMDV/SMDD-SMDV_velocity_f10_25.csv'
    #experiment_name = 'RIB-AVE/RIB-AVE_velocity_f10_54'
    #ds_name = 'RIB-AVE/RIB-AVE_velocity_f10_54.csv'
    model_name = '_CEBRA-behavior.pt'

    '''
    ### model parameters for SMDD-SMDV_velocity_f10_25
    model_parameters = {
                'model_architecture': 'offset10-model',
                'learning_rate': 0.001,
                'min_temperature': 1,
                'num_hidden_units': 32,
                'max_iterations': 10000,
                'device': 'cuda:0',
                'batch_size': 1024,
                'output_dimension': 3
    }
    '''

    ### model parameters for RIB-AVE/RIB-AVE_velocity_f10_54
    model_parameters = {'model_architecture': "offset10-model",
                        'learning_rate': 0.001,
                        'min_temperature': 0.1,
                        'num_hidden_units': 16,
                        'max_iterations': 10000,
                        'device': 'cuda:8',
                        'batch_size': 1024,
                        'output_dimension': 3}
    maxdim = 2
    ripser_path = '/home/alicia/data_personal/ripser_outputs'
    ripser_file = f"ripser_output_{experiment_name.split('/')[1]}_H{maxdim}.json"

    load_start = time.time()
    with open(f'{ripser_path}/{ripser_file}', 'r') as f:
        ripser_output = json.load(f)
    load_end = time.time()
    print(f'load time: {load_end - load_start}')

    num_shuffled_models = 500
    betti_start = time.time()
    betti_number = compute_robust_betti_number(
                                ripser_output,
                                ds_name,
                                model_parameters,
                                num_shuffled_models,
                                maxdim)
    betti_end = time.time()
    print(f'betti time: {betti_end - betti_start}')
    print(f'betti_number: {betti_number}')
    betti_number_file = 'betti_number_' + experiment_name.split('/')[1]
    betti_number_json = json.dumps(betti_number, cls=NumpyEncoder, indent=4)

    with open(f'{ripser_path}/{betti_number_file}.json', 'w') as f:
        f.write(betti_number_json)


if __name__ == "__main__":
    run()

