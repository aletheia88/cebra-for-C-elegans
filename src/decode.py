from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import cebra
import numpy as np


def knn_decode(embedding_train,
           embedding_test,
           label_train,
           label_test,
           num_neighbors):

    decoder = cebra.KNNDecoder(n_neighbors=num_neighbors,
                               metric="cosine")
    decoder.fit(embedding_train, label_train)
    prediction = decoder.predict(embedding_test)

    test_score = r2_score(label_test, prediction)
    test_error = np.linalg.norm(prediction - label_test, 'fro')
    return test_score, test_error


def linear_decode(neural_train,
                  label_train,
                  neural_test,
                  label_test):

    reg = LinearRegression().fit(neural_train, label_train)
    test_avg_error = np.median(abs(reg.predict(neural_test) - label_test))
    test_r2_score = reg.score(neural_test, label_test)
    return test_avg_error, test_r2_score
