from evaluate import evaluate_embeddings, evaluate_single_embedding
from evaluate import iterate
from fit import fit_single_session_cebra, fit_linear_model
from sklearn.metrics.pairwise import cosine_similarity
from util import create_train_test_sets
import cebra
import numpy as np
import os
import torch


def experiment_0():

    '''Decode a single embedding with different K values in KNN and evaluate
    its performance'''

    knn_neighbors = [2*n+1 for n in range(1, 1000, 10)]
    model_path = \
    '/home/alicia/data3_personal/cebra_grid_searches/2023-06-05-17/2023-06-05-17_pumping_f20_subset_0'
    model_name = \
    'min_temperature_0.01_num_hidden_units_16_output_dimension_3_CEBRA-behavior.pt'
    device = torch.device('cuda:3')
    model = cebra.CEBRA.load(f'{model_path}/{model_name}', map_location=device)

    ds_name = '2023-06-05-17/2023-06-05-17_pumping_f20_subset_0.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 10
    noise_multiplier = 1
    train_test_sets = create_train_test_sets(ds_name,
                        noise_ds_name,
                        train_ratio,
                        num_train_test_splits,
                        num_augments,
                        noise_multiplier)

    neural_trains = train_test_sets["neural_trains"]
    label_trains = train_test_sets["label_trains"]
    neural_tests = train_test_sets["neural_tests"]
    label_tests = train_test_sets["label_tests"]
    evaluate_single_embedding(model, knn_neighbors, num_train_test_splits, neural_trains, label_trains,
            neural_tests, label_tests)


def experiment_1():

    '''Run grid search to find the best CEBRA model for the given dataset'''

    ds_date = '2023-06-05-17'
    behavior = 'speed'
    models_path = \
     f'/home/alicia/data3_personal/cebra_grid_searches/{ds_date}/{ds_date}_{behavior}_f20_subset_0'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    device_name = 'cuda:3'
    evaluate_embeddings(models_path, noise_ds_name, KNN_NEIGHBORS, device_name)


def experiment_2():

    '''Fit single-session grid search on concatenated dataset'''

    ds_name = 'SMDD-SMDV/SMDD-SMDV_velocity_f10.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    noise_multiplier = 1
    ds_path = \
        '/home/alicia/data3_personal/cebra_data/SMDD-SMDV/SMDD-SMDV_velocity_f10.csv'
    fit_single_session_cebra('SMDD-SMDV/SMDD-SMDV_velocity_f10.csv',
            num_augments=0, num_label=25)


def experiment_3():

    '''Investigate:
        - if the best SMDD-SMDV-velocity CEBRA fitted on the
            concatenated SMDD-SMDV-velocity dataset predicts velocity well;
            using cosine similarity
        - if there exists high cosine similarity across different datasets'''

    ds_name = 'SMDD-SMDV/SMDD-SMDV_velocity_f10.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    noise_multiplier = 1
    num_augmentations = 0
    num_label = 25
    model_path = \
        '/home/alicia/data3_personal/cebra_grid_searches/SMDD-SMDV/SMDD-SMDV_velocity_f10'
    model_name = 'min_temperature_1_num_hidden_units_16_CEBRA-behavior.pt'
    num_neighbors = 383

    train_test_sets = create_train_test_sets(ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augmentations, noise_multiplier, num_label)
    neural_trains = train_test_sets["neural_trains"]
    label_trains = train_test_sets["label_trains"]
    neural_tests = train_test_sets["neural_tests"]
    label_tests = train_test_sets["label_tests"]
    print(f'label_test shape: {label_tests[0].shape}')

    device = torch.device('cuda:0')
    model = cebra.CEBRA.load(f'{model_path}/{model_name}', map_location=device)

    aggregate_l2_norm = 0
    aggregate_column_cosine_simlarity = 0
    aggregate_dataset_cosine_similarity = np.zeros((num_label, num_label))

    for n in range(num_train_test_splits):
        for i in range(num_label):
            for j in range(num_label):
                dataset_similarity = cosine_similarity(
                            label_tests[n][:, i].reshape(1, -1),
                            label_tests[n][:, j].reshape(1, -1)
                        )
                aggregate_dataset_cosine_similarity[i][j] = \
                    aggregate_dataset_cosine_similarity[i][j] + dataset_similarity
    average_dataset_cosine_similarity = aggregate_dataset_cosine_similarity / num_train_test_splits

    for i in range(num_train_test_splits):
        embedding_train = model.transform(neural_trains[i])
        embedding_test = model.transform(neural_tests[i])
        decoder = cebra.KNNDecoder(n_neighbors=num_neighbors, metric="cosine")
        decoder.fit(embedding_train, label_trains[i])
        prediction = decoder.predict(embedding_test)
        l2_norm = np.linalg.norm(prediction - label_tests[i], 'fro')
        # column_cosine_simlarity.shape = (25, 25)
        column_cosine_simlarity = cosine_similarity(prediction.T, label_tests[i].T)

        #aggregate_column_cosine_simlarity += np.diag(column_cosine_simlarity)
        aggregate_l2_norm += l2_norm
        aggregate_column_cosine_simlarity += column_cosine_simlarity

    average_column_cosine_simlarity = aggregate_column_cosine_simlarity / num_train_test_splits
    average_l2_norm = aggregate_l2_norm / num_train_test_splits
    #print(f'average_column_cosine_simlarity: {average_column_cosine_simlarity}')
    print(f'average_l2_norm: {average_l2_norm}')

    return average_column_cosine_simlarity, average_dataset_cosine_similarity


def experiment_4():

    '''Compute the cosine simlarity value of SMDD-SMDV-velocity CEBRA fitted on
    a single animal SMDD-SMDV-velocity dataset with 10 augmentations'''

    ds_name = 'SMDD-SMDV/SMDD-SMDV_velocity_f10_0.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    noise_multiplier = 1
    num_augmentations = 0
    num_label = 1
    model_path = \
        '/home/alicia/data3_personal/cebra_grid_searches/SMDD-SMDV/SMDD-SMDV_velocity_f10_0'
    model_name = 'min_temperature_1_num_hidden_units_32_CEBRA-behavior.pt'
    num_neighbors = 203

    train_test_sets = create_train_test_sets(ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augmentations, noise_multiplier, num_label)
    neural_trains = train_test_sets["neural_trains"]
    label_trains = train_test_sets["label_trains"]
    neural_tests = train_test_sets["neural_tests"]
    label_tests = train_test_sets["label_tests"]

    device = torch.device('cuda:0')
    model = cebra.CEBRA.load(f'{model_path}/{model_name}', map_location=device)
    aggregate_l2_norm = 0
    aggregate_column_cosine_simlarity = 0

    for i in range(num_train_test_splits):
        embedding_train = model.transform(neural_trains[i])
        embedding_test = model.transform(neural_tests[i])
        decoder = cebra.KNNDecoder(n_neighbors=num_neighbors, metric="cosine")
        decoder.fit(embedding_train, label_trains[i])
        prediction = decoder.predict(embedding_test)
        l2_norm = np.linalg.norm(prediction - label_tests[i], 'fro')
        column_cosine_simlarity = cosine_similarity(prediction.T,
                label_tests[i].T)
        #aggregate_column_cosine_simlarity += np.diag(column_cosine_simlarity)
        aggregate_l2_norm += l2_norm
        aggregate_column_cosine_simlarity += column_cosine_simlarity

    average_column_cosine_simlarity = aggregate_column_cosine_simlarity / num_train_test_splits
    average_l2_norm = aggregate_l2_norm / num_train_test_splits
    print(f'average_column_cosine_simlarity: {average_column_cosine_simlarity}')
    print(f'average_l2_norm: {average_l2_norm}')

    return average_column_cosine_simlarity


def experiment_5():

    '''Fit single-session CEBRA on concatenated dataset
    `SMDD-SMDV_velocity_f10_25.csv`'''

    parameter_grid = dict(
        model_architecture="offset10-model",
        min_temperature=1,
        #min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=[10],
        max_iterations=[50000],
        learning_rate=[0.001],
        output_dimension=[3],
        num_hidden_units=32,
        #num_hidden_units=[16, 32],
        batch_size=1024,
        device='cuda:0',
        verbose=True)
    ds_name = 'SMDD-SMDV/SMDD-SMDV_velocity_f10_25.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 0
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 300, 10)]
    num_label = 1
    save_models_dir = f"/home/alicia/data3_personal/cebra_grid_searches/{ds_name.split('.')[0]}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def experiment_5_1():

    '''Fit single-session CEBRA on concatenated dataset
    `RIB-AVE_velocity_f10_54.csv`'''

    parameter_grid = dict(
        model_architecture="offset10-model",
        #min_temperature=[0.01, 0.1, 1],
        min_temperature=0.1,
        temperature_mode="auto",
        time_offsets=[10],
        max_iterations=[50000],
        learning_rate=[0.001],
        output_dimension=[3],
        num_hidden_units=16,
        #num_hidden_units=[16, 32],
        batch_size=1024,
        device='cuda:0',
        verbose=True)
    ds_name = 'RIB-AVE/RIB-AVE_velocity_f10_54.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 0
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 300, 10)]
    num_label = 1
    save_models_dir = f"/home/alicia/data3_personal/cebra_grid_searches/{ds_name.split('.')[0]}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def experiment_6():

    '''Fit single-session CEBRA on a single-animal dataset
    `RIB-AVE_velocity_fz_1.csv`'''

    parameter_grid = dict(
        min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=[10],
        max_iterations=[20000],
        learning_rate=[0.0001],
        output_dimension=[3],
        num_hidden_units=[16, 32],
        batch_size=None,
        device='cuda:1',
        verbose=True)
    ds_name = 'RIB-AVE/RIB-AVE_velocity_fz_1.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 10
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 200, 10)]
    num_label = 1
    save_models_dir = f"/home/alicia/data3_personal/cebra_grid_searches/{ds_name.split('.')[0]}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def experiment_7():

    '''Fit single-session CEBRA on a single-animal dataset
    `RIB-AVE_velocity_fcube_1.csv`'''

    parameter_grid = dict(
        min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=[10],
        max_iterations=[20000],
        learning_rate=[0.0001],
        output_dimension=[3],
        num_hidden_units=[16, 32],
        batch_size=None,
        device='cuda:0',
        verbose=True)
    ds_name = 'RIB-AVE/RIB-AVE_velocity_fcube_1.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 10
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 200, 10)]
    num_label = 1
    save_models_dir = f"/home/alicia/data3_personal/cebra_grid_searches/{ds_name.split('.')[0]}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def experiment_8():

    '''Fit single-session CEBRA on a single-animal dataset
    `RIB-AVE_velocity_f10_1.csv`'''

    parameter_grid = dict(
        model_architecture="offset10-model",
        min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=[10],
        max_iterations=[20000],
        learning_rate=[0.001],
        output_dimension=[3],
        num_hidden_units=[16, 32],
        batch_size=None,
        device='cuda:0',
        verbose=True)

    ds_name = 'RIB-AVE/RIB-AVE_velocity_f10_1.csv'
    noise_ds_name = '2022-01-07-03/2022-01-07-03_F20.csv'
    train_ratio = 0.8
    num_train_test_splits = 5
    num_augments = 10
    noise_multiplier = 1
    num_neighbors = [2*n+1 for n in range(1, 200, 10)]
    num_label = 1
    save_models_dir = f"/home/alicia/data3_personal/cebra_grid_searches/{ds_name.split('.')[0]}"
    if not os.path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    iterate(parameter_grid, ds_name, noise_ds_name, train_ratio,
            num_train_test_splits, num_augments, noise_multiplier,
            num_neighbors, num_label, save_models_dir)


def experiment_9():

    '''Fit the linear-model counterparts for the cebra models above'''

    #ds_name1 = 'RIB-AVE/RIB-AVE_velocity_f10_54.csv'
    #ds_name2 = 'SMDD-SMDV/SMDD-SMDV_velocity_f10_25.csv'
    ds_name3 = 'RIB-AVE/RIB-AVE_velocity_f10_1.csv'
    ds_name4 = 'RIB-AVE/RIB-AVE_velocity_fcube_1.csv'
    ds_name5 = 'RIB-AVE/RIB-AVE_velocity_fz_1.csv'

    #r2_1 = fit_linear_model(ds_name1)
    #r2_2 = fit_linear_model(ds_name2)
    r2_3 = fit_linear_model(ds_name3)
    r2_4 = fit_linear_model(ds_name4)
    r2_5 = fit_linear_model(ds_name5)

    #print(f'{ds_name1}: {r2_1}')
    #print(f'{ds_name2}: {r2_2}')
    print(f'RIB-AVE_velocity_f10_1: {r2_3}')
    print(f'RIB-AVE_velocity_fcube_1: {r2_4}')
    print(f'RIB-AVE_velocity_fz_1: {r2_5}')


if __name__ == "__main__":
    experiment_9()
