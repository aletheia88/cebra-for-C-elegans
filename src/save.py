from fit import preprocess
from util import create_train_test_sets, get_matrix_derivative, augment
import argparse
import cebra
import h5py
import json
import numpy as np
import os
import pandas as pd


def save_multi_session_model_outputs(parameters):

    model_path = f'/home/alicia/data_personal/cebra_outputs/RIB-AVE'
    model_name = \
    f'learning_rate_{parameters[0]}_min_temperature_{parameters[1]}_output_dimension_{parameters[2]}_CEBRA-behavior.pt'
    model = cebra.CEBRA.load(f'{model_path}/{model_name}')

    neurons = 'RIB-AVE'
    ds_path = f'/home/alicia/data_personal/cebra_data/{neurons}'
    neural_df = pd.read_csv(f'{ds_path}/{neurons}_f10_0.csv')
    neural_data = neural_df.to_numpy()
    neural_data_series = preprocess(neural_data)
    ds_names = neural_df.columns.tolist()

    save_dir = f'/home/alicia/data_personal/cebra_outputs/{neurons}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (name, X) in enumerate(zip(ds_names, neural_data_series)):
        embedding = model.transform(X, session_id=i)
        file = h5py.File(f"{save_dir}/{model_name.split('.p')[0]}_embd{i}.h5", 'w')
        dataset = file.create_dataset('embedding', data=embedding)
        print("model's test embedding saved!")
        file.close()


def save_augmented_model_outputs(experiment_name, model_name, ds_name,
            save_augmented_dataset=False):

    model_path = f'/home/alicia/data_personal/cebra_grid_searches/{experiment_name}'
    model = cebra.CEBRA.load(f'{model_path}/{model_name}')
    ds_path = '/home/alicia/data_personal/cebra_data'
    df = pd.read_csv(f'{ds_path}/{ds_name}')
    neural_data = df.iloc[:, :-1].values
    behaviors = df.iloc[:, -1].values.reshape(-1, 1)

    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augmentations = 10
    noise_multiplier = 1
    num_label = 1
    print(neural_data.shape)
    print(behaviors.shape)
    augmented_neural_data, augmented_behaviors = augment(
                                  neural_data,
                                  behaviors,
                                  noise_ds_name,
                                  noise_multiplier,
                                  num_label,
                                  num_augmentations)
    # write to disk
    if save_augmented_dataset:
        new_df = pd.DataFrame(list(df.columns))
        new_df[list(df.columns)[:4]] = augmented_neural_data[:, :4]
        new_df[list(df.columns)[-1]] = augmented_neural_data[:, -1]
        print(new_df)
    '''
    save_dir = f'/home/alicia/data_personal/cebra_outputs/{ds_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    augmented_ds_name = f"{ds_name.split('/')[1]}_aug{num_augmentations}_label.npz"
    np.savez(f'{save_dir}/{augmented_ds_name}', numpy_array=label_test)
    print('labels of the test data saved!')

    losses = model.state_dict_["loss"]
    np.savez(f"{save_dir}/{model_name.split('.p')[0]}_loss.npz",
            numpy_array=losses.numpy())
    print("model's losses saved!")

    temperature = model.state_dict_["log"]["temperature"]
    np.savez(f"{save_dir}/{model_name.split('.p')[0]}_temp.npz",
            numpy_array=np.array(temperature))
    print("model's temperature saved!")

    file = h5py.File(f"{save_dir}/{model_name.split('.p')[0]}_embd.h5", 'w')
    embedding = model.transform(neural_test)
    dataset = file.create_dataset('embedding', data=embedding)
    print("model's test embedding saved!")
    file.close()
    '''

def save_model_outputs(experiment_name, model_name, ds_name):

    '''Saves the given cebra model training loss and embedding (embeddings are
    regenerated from the datasets the model was fit on)

    Args:
        experiment_name: name of the experiment; e.g. 'RIB-AVE_velocity_f10_25'
            meaning the datasets contains RIB-AVE neuronal traces concatenated
            from 25 animal datasets; in each dataset, neuronal traces are
            normalized by dividing the 10th percentile flurescence as the
            baseline, and animal velocity across time
        model_name: name of the best model; e.g.
            'min_temperature_0.1_num_hidden_units_16_CEBRA-behavior_k323'
    '''

    model_search_path = '/home/alicia/data3_personal/cebra_grid_searches'
    ds_path = '/home/alicia/data3_personal/cebra_data'
    model = cebra.CEBRA.load(f'{model_search_path}/{experiment_name}/{model_name}')
    df = pd.read_csv(f'{ds_path}/{ds_name}')
    print(df.iloc[:, :-2])
    neural_data = df.iloc[:, :-2].values

    # add derivative columns to neural traces
    derivatives = get_matrix_derivative(neural_data)
    neural_data_n_derivatives = np.hstack((neural_data, derivatives))

    embedding = model.transform(neural_data_n_derivatives)

    model_outputs_path = f'/home/alicia/data3_personal/cebra_outputs/{experiment_name}'
    if not os.path.exists(model_outputs_path):
        os.mkdir(model_outputs_path)

    losses = model.state_dict_["loss"]
    np.savez(f"{model_outputs_path}/{model_name.split('_C')[0]}_loss.npz",
            numpy_array=losses.numpy())
    print("model loss saved!")

    temperature = model.state_dict_["log"]["temperature"]
    np.savez(f"{model_outputs_path}/{model_name.split('_C')[0]}_temp.npz",
            numpy_array=np.array(temperature))
    print("model temperature saved!")

    file = h5py.File(f"{model_outputs_path}/{model_name.split('_C')[0]}_embd.h5", 'w')
    dataset = file.create_dataset('embedding', data=embedding)
    print("model embedding saved!")
    file.close()


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-mn",
                        metavar="STRING",
                        help="Enter the model name")
    parser.add_argument("--ds_name", "-dn",
                        metavar="STRING",
                        help="Enter the dataset name")
    args = parser.parse_args()
    save_model_outputs(args)
    '''
    ds_name = 'RIB-AVE-AVA-X/RIB-AVE-AVA-X_turndiff_f10_1.csv'
    experiment_name = 'RIB-AVE-AVA-X/RIB-AVE-AVA-X_turndiff_f10_1'
    model_name = \
        'learning_rate_0.0001_min_temperature_0.01_num_hidden_units_8_output_dimension_5_CEBRA-behavior.pt'
    #save_augmented_model_outputs(experiment_name, model_name, ds_name, True)
    save_model_outputs(experiment_name, model_name, ds_name)

    #ds_name = '2022-06-14-13/2022-06-14-13_velocity_f20_subset_0'
    #parameters = [0.001, 0.1, 3]
    #save_augmented_model_outputs(ds_name, parameters)
    #save_multi_session_model_outputs(parameters)
