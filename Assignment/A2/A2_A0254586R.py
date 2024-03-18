import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0254586R(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    pass

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = N)
    
    encoder = OneHotEncoder()
    Ytr = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    Yts = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = np.zeros(8)
    error_test_array = np.zeros(8)  

    for order in range(1,9):
        poly = PolynomialFeatures(order)
        P_train = poly.fit_transform(X_train)
        P_test = poly.fit_transform(X_test)
        Ptrain_list.append(P_train)
        Ptest_list.append(P_test)

        if P_train.shape[0] <= P_train.shape[1]:
            reg_L2 = 0.0001 * np.identity(P_train.shape[0])
            w_dual_ridge = P_train.T @ (inv(P_train @ P_train.T + reg_L2)) @ Ytr
            w_list.append(w_dual_ridge)
        else:
            reg_L = 0.0001 * np.identity(P_train.shape[1])
            w_primal_ridge = inv(P_train.T @ P_train + reg_L) @ P_train.T @ Ytr
            w_list.append(w_primal_ridge)

        y_train_pred = np.argmax(np.dot(P_train, w_list[order - 1]), axis = 1)
        error_train_array[order - 1] = np.sum(y_train_pred != y_train)
        
        y_test_pred = np.argmax(np.dot(P_test, w_list[order - 1]), axis = 1)
        error_test_array[order - 1] = np.sum(y_test_pred != y_test)

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array