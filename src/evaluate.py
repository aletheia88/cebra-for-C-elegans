from collections import Counter
from decode import knn_decode, linear_decode
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from util import create_train_test_sets
import cebra
import json
import numpy as np
import re
import scipy.stats as stats


GCaMP_models_dir = "/home/alicia/data3_personal/cebra_grid_searches/gcamp_augment"
GFP_models_dir = "/home/alicia/data3_personal/cebra_grid_searches/gfp_augment"
score_dir = "/home/alicia/data3_personal/cebra_eval"

PARAMETER_GRID = dict(
        min_temperature=[0.01, 0.1, 1],
        temperature_mode = "auto",
        time_offsets=[10],
        max_iterations=[30000],
        learning_rate=[0.0001],
        output_dimension=[3],
        num_hidden_units=[16, 32],
        device='cuda:1',
        verbose=True)

noise_multipliers = sorted([n for n in range(1, 11)] + \
        [float("{:.2f}".format(1 + 1/n)) for n in range(2, 7)])


def iterate(
            parameter_grid,
            ds_name,
            noise_ds_name,
            train_ratio,
            num_train_test_splits,
            num_augments,
            noise_multiplier,
            num_neighbors,
            num_label,
            save_models_dir):
    """Function that performs a grid search over the indicated parameter space
    to record the decoding performance for each combination of the model
    parameters; the search is performed with fixed training data treatment,
    IOW, a fixed number of augmentation and fixed noise multiplier
    """

    aggregate_test_score = dict()
    aggregate_test_error = dict()

    train_test_sets = create_train_test_sets(ds_name,
                                    noise_ds_name,
                                    train_ratio,
                                    num_train_test_splits,
                                    num_augments,
                                    noise_multiplier,
                                    num_label)
    neural_trains = train_test_sets["neural_trains"]
    label_trains = train_test_sets["label_trains"]
    neural_tests = train_test_sets["neural_tests"]
    label_tests = train_test_sets["label_tests"]

    for i in range(num_train_test_splits):
        test_scores, test_errors = grid_search(
                                    parameter_grid,
                                    neural_trains[i],
                                    label_trains[i],
                                    neural_tests[i],
                                    label_tests[i],
                                    num_neighbors,
                                    num_augments,
                                    save_models_dir
                                    )
        aggregate_test_score = dict(Counter(aggregate_test_score) + \
                                    Counter(test_scores))
        aggregate_test_error = dict(Counter(aggregate_test_error) + \
                                    Counter(test_errors))

    average_test_score = {model_name: test_score/num_train_test_splits for model_name,
            test_score in aggregate_test_score.items()}
    average_test_error = {model_name: test_error/num_train_test_splits for model_name,
            test_error in aggregate_test_error.items()}
    high_test_score_models, low_test_error_models = get_best_models(
                                average_test_score,
                                average_test_error
                                )
    average_test_score["high_test_score_models"] = high_test_score_models
    average_test_error["low_test_error_models"] = low_test_error_models

    with open(f"{save_models_dir}/average_test_score.json", 'w') as jf:
        json.dump(average_test_score, jf, indent=4)
    with open(f"{save_models_dir}/average_test_error.json", 'w') as jf:
        json.dump(average_test_error, jf, indent=4)


def grid_search(parameter_grid,
                neural_train,
                label_train,
                neural_test,
                label_test,
                num_neighbors,
                num_augments,
                save_models_dir):

    datasets = {
            # "CEBRA-time": neural_train, # time contrastive learning
            "CEBRA-behavior": (neural_train, label_train) # behavioral contrastive
            }
    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets,
                           params=parameter_grid,
                           models_dir=save_models_dir)

    model_names = grid_search.models_names
    # load the embedding from the save models
    embeddings_train = \
                [cebra.CEBRA.load(f"{save_models_dir}/{model_name}.pt").transform(neural_train)
                for model_name in model_names]
    embeddings_test = \
                [cebra.CEBRA.load(f"{save_models_dir}/{model_name}.pt").transform(neural_test)
                for model_name in model_names]

    test_scores = dict()
    test_errors = dict()

    # iteratively decode and write the decoding accuracy to a dataframe
    for i in range(len(model_names)):
        for k in num_neighbors:
            test_score, test_error = knn_decode(embeddings_train[i],
                                            embeddings_test[i],
                                            label_train,
                                            label_test,
                                            k)
            # store scores in dict
            test_scores[f"{model_names[i]}_k{k}"] = test_score
            test_errors[f"{model_names[i]}_k{k}"] = test_error

    return test_scores, test_errors


def get_best_models(test_score_dict, test_error_dict, num_bests=3):

    # models with test scores from highest to lowest
    test_scores_sorted = [key for key, value in sorted(test_score_dict.items(),
                                                key=lambda item: item[1],
                                                reverse=True)]
    # models with test errors from lowest to highest
    test_errors_sorted = [key for key, value in sorted(test_error_dict.items(),
                                                key=lambda item: item[1])]

    return test_scores_sorted[:num_bests], test_errors_sorted[:num_bests]


def evaluate_embeddings(models_path, noise_ds_name, knn_neighbors, device_name,
                      train_ratio=0.8, num_train_test_splits=5,
                      num_augments=10, noise_multiplier=1):

    '''Decode embeddings using the given model that has already been trained on
    some train and test set'''

    device = torch.device(device_name)
    items = os.listdir(models_path)
    model_names = [item for item in items if not item.endswith('json') and
                   not item.endswith('pkl')]
    ds_name = os.path.join(models_path.split('cebra_grid_searches')[1].lstrip('/') + '.csv')

    train_test_sets = create_train_test_sets(
                        ds_name,
                        noise_ds_name,
                        train_ratio,
                        num_train_test_splits,
                        num_augments,
                        noise_multiplier)
    neural_trains = train_test_sets["neural_trains"]
    label_trains = train_test_sets["label_trains"]
    neural_tests = train_test_sets["neural_tests"]
    label_tests = train_test_sets["label_tests"]

    model_r2_dict = dict()

    for model_name in tqdm(model_names):
        model = cebra.CEBRA.load(f'{models_path}/{model_name}',
                map_location=device)
        # `models_path` is given in the form of
        # `~/data3_personal/cebra_grid_searches/<yyyy-mm-dd-id>/<ds_name>`
        average_test_scores = evaluate_single_embedding(
                        model,
                        knn_neighbors,
                        num_train_test_splits,
                        neural_trains,
                        label_trains,
                        neural_tests,
                        label_tests
                    )
        max_test_score = max(average_test_scores)
        optimal_k = knn_neighbors[average_test_scores.index(max_test_score)]
        model_r2_dict[f'{model_name}_k{optimal_k}'] = max_test_score

    with open(f'{models_path}/average_test_score_v1.json', 'w') as f:
        json.dump(model_r2_dict, f, indent=4)


def evaluate_single_embedding(model, knn_neighbors, num_train_test_splits,
        neural_trains, label_trains, neural_tests, label_tests):

    '''Function returns the average test scores in the order of the value of K
    that is used in KNN clustering'''

    average_test_scores = []

    for k in knn_neighbors:
        average_test_score = 0
        for i in range(num_train_test_splits):
            embedding_train = model.transform(neural_trains[i])
            embedding_test = model.transform(neural_tests[i])
            test_score, _ = knn_decode(
                    embedding_train,
                    embedding_test,
                    label_trains[i],
                    label_tests[i],
                    k)
            average_test_score += test_score
        average_test_scores.append(average_test_score/num_train_test_splits)
        print(f'k={k}: r^2={average_test_score/num_train_test_splits}')
    return average_test_scores


#############################################
##### Previous (june, july) evaluations #####
#############################################


def do_pairwise_t_test(GFP_ds_name,
                       noise_ds_name,
                       train_ratio,
                       num_train_test_splits,
                       num_augment,
                       noise_multiplier):
    """Function that computes the T-statistics of the Mean Square Error of the
    null model vs. the linear model on GFP data; the goal is to see if there
    exists a statistically significant difference in the two models'
    performance on predicting behaviral variables
    """

    train_test_sets = create_train_test_sets(
                                GFP_ds_name,
                                noise_ds_name,
                                train_ratio,
                                num_train_test_splits,
                                num_augment,
                                noise_multiplier)
    # H0: mean of MSE(GFP null) equal to mean of MSE(GFP linear)
    # Ha: mean of MSE(GFP null) not equal to mean of MSE(GFP linear)
    # n = 5
    # with diffferent noise multipliers
    GFP_neural_trains = train_test_sets["neural_trains"]
    GFP_label_trains = train_test_sets["label_trains"]
    GFP_neural_tests = train_test_sets["neural_tests"]
    GFP_label_tests = train_test_sets["label_tests"]

    MSE_GFP_null_model = []
    MSE_GFP_linear_model = []

    for i in range(num_train_test_splits):
        # compute MSE of the null model on GFP
        GFP_dimension = GFP_label_tests[i].shape[0]
        GFP_null_model_predicted_label = np.array(
                            GFP_dimension * [np.mean(GFP_label_trains[i])])
        MSE_GFP_null_model.append(
                            mean_squared_error(
                                GFP_label_tests[i],
                                GFP_null_model_predicted_label)
                            )
        # compute MSE of linear decoder on GFP
        GFP_reg = LinearRegression().fit(GFP_neural_trains[i],
                                         GFP_label_trains[i])
        MSE_GFP_linear_model.append(
                            mean_squared_error(
                                GFP_reg.predict(GFP_neural_tests[i]),
                                GFP_label_tests[i])
                            )
    print(f"noise multiplier = {noise_multiplier}")
    t_test_result = stats.ttest_rel(MSE_GFP_null_model, MSE_GFP_linear_model)
    print(t_test_result)


def get_pseudo_r2_for_all_noise_multiplier(
                                GCaMP_ds_name,
                                GFP_ds_name,
                                noise_ds_name,
                                train_ratio,
                                num_train_test_splits,
                                num_augment):
    """Function that computes the pseudo-R^2 values for the indicated models
    (e.g. linear model on GCaMP/GFP data, KNN decoder on CEBRA-trained embedding
    using GCaMP/GFP data) from varying the noise multiplier
    """

    pseudo_r2_dict = {"GCaMP_knn": [],
                      "GCaMP_linear": [],
                      "GFP_knn": [],
                      "GFP_linear": []
                     }

    for noise_multiplier in tqdm(noise_multipliers):
        pseudo_r2s = get_pseudo_r2(GCaMP_ds_name,
                      GFP_ds_name,
                      noise_ds_name,
                      train_ratio,
                      num_train_test_splits,
                      num_augment,
                      noise_multiplier)

        for decoder, r2_value in pseudo_r2s.items():
            pseudo_r2_dict[decoder].append(r2_value)

    with open(f"{models_dir}pseudo_r2_vs_noise_multiplier_not_averaged.json", "w") as jf:
        json.dump(pseudo_r2_dict, jf, indent=4)


def get_pseudo_r2_for_all_augment(GCaMP_ds_name,
                                  GFP_ds_name,
                                  noise_ds_name,
                                  train_ratio,
                                  num_train_test_splits,
                                  num_augments):
    """Function that computes the pseudo-R^2 values for the indicated models
    (e.g. linear model on GCaMP/GFP data, KNN decoder on CEBRA-trained embedding
    using GCaMP/GFP data) from varying the number of times augmenting the
    original data by adding Gaussian noise
    """

    pseudo_r2_dict = {"GCaMP_knn": [],
                      "GCaMP_linear": [],
                      "GFP_knn": [],
                      "GFP_linear": []
                     }

    for num_augment in tqdm(range(num_augments)):
        pseudo_r2s = get_pseudo_r2(GCaMP_ds_name,
                      GFP_ds_name,
                      noise_ds_name,
                      train_ratio,
                      num_train_test_splits,
                      num_augment)
        for decoder, r2_value in pseudo_r2s.items():
            pseudo_r2_dict[decoder].append(r2_value)

    with open(f"{models_dir}pseudo_r2.json", "w") as jf:
        json.dump(pseudo_r2_dict, jf, indent=4)


def get_pseudo_r2(GCaMP_ds_name,
                  GFP_ds_name,
                  noise_ds_name,
                  train_ratio,
                  num_train_test_splits,
                  num_augment,
                  noise_multiplier=1):
    """Function that computes the pseudo-R^2 values for the indicated models
    (e.g. linear model on GCaMP/GFP data, KNN decoder on CEBRA-trained embedding
    using GCaMP/GFP data) with the given treatment of training data (i.e.number
    of augmentations, noise multiplier size)
    """

    # create train-test splits
    GCaMP_train_test_sets = create_train_test_sets(
                                    GCaMP_ds_name,
                                    noise_ds_name,
                                    train_ratio,
                                    num_train_test_splits,
                                    num_augment,
                                    noise_multiplier)

    GFP_train_test_sets = create_train_test_sets(
                                    GFP_ds_name,
                                    noise_ds_name,
                                    train_ratio,
                                    num_train_test_splits,
                                    num_augment,
                                    noise_multiplier)

    GCaMP_neural_trains = GCaMP_train_test_sets["neural_trains"]
    GCaMP_label_trains = GCaMP_train_test_sets["label_trains"]
    GCaMP_neural_tests = GCaMP_train_test_sets["neural_tests"]
    GCaMP_label_tests = GCaMP_train_test_sets["label_tests"]

    GFP_neural_trains = GFP_train_test_sets["neural_trains"]
    GFP_label_trains = GFP_train_test_sets["label_trains"]
    GFP_neural_tests = GFP_train_test_sets["neural_tests"]
    GFP_label_tests = GFP_train_test_sets["label_tests"]

    GCaMP_null_model_MSEs_per_augment = []
    GCaMP_knn_decoder_MSEs_per_augment = []
    GCaMP_linear_decoder_MSEs_per_augment = []

    GFP_null_model_MSEs_per_augment = []
    GFP_knn_decoder_MSEs_per_augment = []
    GFP_linear_decoder_MSEs_per_augment = []

    # load the best CEBRA model trained on GCaMP data
    with open (f"{GCaMP_models_dir}{num_augment}/average_test_score.json", 'r') as jf:
        content = jf.read()
        GCaMP_score_dict = json.loads(content)

    GCaMP_model_params = GCaMP_score_dict["high_test_score_models"][0]
    GCaMP_k_neighbor = int(re.findall(r'k(\d+)', GCaMP_model_params)[0])
    best_GCaMP_model_name = re.sub(r'_k\d+', '', GCaMP_model_params)
    best_GCaMP_model = \
                cebra.CEBRA.load(f"{GCaMP_models_dir}{num_augment}/{best_GCaMP_model_name}.pt")

    # load the best CEBRA model trained on GFP data
    with open (f"{GFP_models_dir}{num_augment}/average_test_score.json", 'r') as jf:
        content = jf.read()
        GFP_score_dict = json.loads(content)

    GFP_model_params = GFP_score_dict["high_test_score_models"][0]
    GFP_k_neighbor = int(re.findall(r'k(\d+)', GFP_model_params)[0])
    best_GFP_model_name = re.sub(r'_k\d+', '', GFP_model_params)
    best_GFP_model = \
                cebra.CEBRA.load(f"{GFP_models_dir}{num_augment}/{best_GFP_model_name}.pt")


    for i in range(num_train_test_splits):

        # compute MSE of the null model on GCaMP
        GCaMP_dimension = GCaMP_label_tests[i].shape[0]
        GCaMP_null_model_predicted_label = np.array(
                            GCaMP_dimension * [np.mean(GCaMP_label_trains[i])])
        GCaMP_null_model_MSEs_per_augment.append(
                            mean_squared_error(
                                GCaMP_label_tests[i],
                                GCaMP_null_model_predicted_label)
                            )

        # compute MSE of the null model on GFP
        GFP_dimension = GFP_label_tests[i].shape[0]
        GFP_null_model_predicted_label = np.array(
                            GFP_dimension * [np.mean(GFP_label_trains[i])])
        GFP_null_model_MSEs_per_augment.append(
                            mean_squared_error(
                                GFP_label_tests[i],
                                GFP_null_model_predicted_label)
                            )

        # compute MSE of the knn decoder on GCaMP
        embedding_GCaMP_train = best_GCaMP_model.transform(GCaMP_neural_trains[i])
        embedding_GCaMP_test = best_GCaMP_model.transform(GCaMP_neural_tests[i])
        GCaMP_knn_decoder = cebra.KNNDecoder(GCaMP_k_neighbor, metric="cosine")
        GCaMP_knn_decoder.fit(embedding_GCaMP_train, GCaMP_label_trains[i])
        GCaMP_knn_predicted_label = GCaMP_knn_decoder.predict(embedding_GCaMP_test)
        GCaMP_knn_decoder_MSEs_per_augment.append(
                            mean_squared_error(
                                GCaMP_knn_predicted_label,
                                GCaMP_label_tests[i])
                            )

        # compute MSE of the knn decoder on GFP
        embedding_GFP_train = best_GFP_model.transform(GFP_neural_trains[i])
        embedding_GFP_test = best_GFP_model.transform(GFP_neural_tests[i])
        GFP_knn_decoder = cebra.KNNDecoder(GFP_k_neighbor, metric="cosine")
        GFP_knn_decoder.fit(embedding_GFP_train, GFP_label_trains[i])
        GFP_knn_predicted_label = GFP_knn_decoder.predict(embedding_GFP_test)
        GFP_knn_decoder_MSEs_per_augment.append(
                            mean_squared_error(
                                GFP_knn_predicted_label,
                                GFP_label_tests[i])
                            )

        # compute MSE of linear decoder on GCaMP
        GCaMP_reg = LinearRegression().fit(GCaMP_neural_trains[i],
                                     GCaMP_label_trains[i])
        GCaMP_linear_decoder_MSEs_per_augment.append(
                            mean_squared_error(
                                GCaMP_reg.predict(GCaMP_neural_tests[i]),
                                GCaMP_label_tests[i])
                            )
        # compute MSE of linear decoder on GFP
        GFP_reg = LinearRegression().fit(GFP_neural_trains[i],
                                         GFP_label_trains[i])
        GFP_linear_decoder_MSEs_per_augment.append(
                            mean_squared_error(
                                GFP_reg.predict(GFP_neural_tests[i]),
                                GFP_label_tests[i])
                            )

    null_model_GCaMP_MSE = np.mean(GCaMP_null_model_MSEs_per_augment)
    null_model_GFP_MSE = np.mean(GFP_null_model_MSEs_per_augment)

    GCaMP_knn_pseudo_r2 = 1 - \
            np.mean(GCaMP_knn_decoder_MSEs_per_augment)/null_model_GCaMP_MSE

    GCaMP_linear_pseudo_r2 = 1 - \
            np.mean(GCaMP_linear_decoder_MSEs_per_augment)/null_model_GCaMP_MSE

    GFP_knn_pseudo_r2 = 1 - \
            np.mean(GFP_knn_decoder_MSEs_per_augment)/null_model_GFP_MSE

    GFP_linear_pseudo_r2 = 1 - \
            np.mean(GFP_linear_decoder_MSEs_per_augment)/null_model_GFP_MSE

    # return the pseudo R^2 obtained from each train-test split
    return {"GCaMP_knn": GCaMP_knn_pseudo_r2,
            "GCaMP_linear": GCaMP_linear_pseudo_r2,
            "GFP_knn": GFP_knn_pseudo_r2,
            "GFP_linear": GFP_linear_pseudo_r2
            }


def evaluate_null_model(ds_name,
                        train_ratio,
                        num_train_test_splits,
                        num_augments):
    """Function that computes the pseudo-R^2 scores of the null model
    performance by varying the number of augmentations
    """

    num_model_MSEs = []
    for num_augment in range(num_augments):
        train_test_sets = create_train_test_sets(ds_name,
                                        noise_ds_name,
                                        train_ratio,
                                        num_train_test_splits,
                                        num_augment)
        neural_trains = train_test_sets["neural_trains"]
        label_trains = train_test_sets["label_trains"]
        neural_tests = train_test_sets["neural_tests"]
        label_tests = train_test_sets["label_tests"]

        predicted_labels = []
        null_model_MSEs_per_augment = []

        for i in range(num_train_test_splits):
            # compute mean behavior value in train
            # set it as the predicted behavior value on test data
            dimension = label_tests[i].shape[0]
            predicted_label = np.array(dimension * [np.mean(label_trains[i])])
            predicted_labels.append(predicted_label)
            # compute MSE of this prediction
            null_model_MSEs_per_augment.append(mean_squared_error(label_tests[i],
                            predicted_label))

        num_model_MSEs.append(np.mean(null_model_MSEs_per_augment))

    return num_model_MSEs


def evaluate_linear_model(ds_name,
                          noise_ds_name,
                          train_ratio,
                          num_train_test_splits,
                          num_augments,
                          noise_multipler=1):
    """Function that computes the pseudo-R^2 scores of the linear model
    performance by varying the number of augmentations or the noise multiplier
    """

    mean_test_score = dict()
    mean_test_error = dict()

    for num_augment in tqdm(range(num_augments)):

        # R^2 of the linear regression
        aggregate_test_score = 0
        # average difference between the model prediction and true test labels
        aggregate_test_error = 0

        train_test_sets = create_train_test_sets(ds_name,
                                        noise_ds_name,
                                        train_ratio,
                                        num_train_test_splits,
                                        num_augment,
                                        noise_multiplier)
        neural_trains = train_test_sets["neural_trains"]
        label_trains = train_test_sets["label_trains"]
        neural_tests = train_test_sets["neural_tests"]
        label_tests = train_test_sets["label_tests"]

        for i in range(num_train_test_splits):
            test_error, test_r2_score = linear_decode(
                                                neural_trains[i],
                                                label_trains[i],
                                                neural_tests[i],
                                                label_tests[i])
            aggregate_test_score += test_r2_score
            aggregate_test_error += test_error

        mean_test_score[f"augment{num_augment}_r2_score"] = \
                aggregate_test_score / num_train_test_splits
        mean_test_error[f"augment{num_augment}_error"] = \
                aggregate_test_error / num_train_test_splits

    with open(f"{score_dir}/GFP_linear_model_avg_r2.json", 'w') as jf:
        json.dump(mean_test_score, jf, indent=4)

    with open(f"{score_dir}/GFP_linear_model_avg_test_error.json", 'w') as jf:
        json.dump(mean_test_error, jf, indent=4)


if __name__ == "__main__":
    ds_name = "2022-06-14-13_Velocity_F20.csv"
    noise_ds_name = "2022-01-07-03_F20.csv"
    GCaMP_ds_name = "2022-06-14-13_Velocity_F20.csv"
    GFP_ds_name = "2022-01-07-03_F20.csv"

    train_ratio = 0.8
    num_train_test_splits = 5
    num_neighbors = [2*n+1 for n in range(1, 100, 10)]
    num_augments = 5
    noise_multiplier = 1

    # evaluate CEBRA models under the given setting
    save_models_dir = f'{GFP_models_dir}{num_augments}'
    '''
    iterate(GFP_ds_name,
            noise_ds_name,
            train_ratio,
            num_train_test_splits,
            num_augments,
            noise_multiplier,
            num_neighbors,
            save_models_dir)
    '''
    # evaluate linear model from the given setting
    evaluate_linear_model(GFP_ds_name,
                          noise_ds_name,
                          train_ratio,
                          num_train_test_splits,
                          num_augments,
                          noise_multiplier)
    # do pair-wise T-test
    """
    for noise_multiplier in noise_multipliers:
        do_pairwise_t_test(GFP_ds_name,
                           noise_ds_name,
                           train_ratio,
                           num_train_test_splits,
                           num_augment,
                           noise_multiplier)
    """
    # evaluate null model
    """
    evaluate_null_model(ds_name, train_ratio, num_train_test_splits,
                        num_augments)
    """
    """
    get_pseudo_r2_for_all_noise_multiplier(
                                GCaMP_ds_name,
                                GFP_ds_name,
                                noise_ds_name,
                                train_ratio,
                                num_train_test_splits,
                                num_augment)
    """
    """
    get_pseudo_r2_for_all_augment(
                  GCaMP_ds_name,
                  GFP_ds_name,
                  noise_ds_name,
                  train_ratio,
                  num_train_test_splits,
                  num_augments)
    """
