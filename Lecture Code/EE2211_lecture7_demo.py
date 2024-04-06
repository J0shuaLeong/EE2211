# # Lecture 7 Demo 1 Polynominal regression with regularization

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.linalg import inv

def CreateRegressors(x, max_order):

    # x is assumed to be array of length N
    # return P = list of regressors based on max_order
    # P[i] are regressors for order i+1 and is of size N x (order+1), where
    # N is number of data points

    P = [] #initialize empty list
    for order in range(1, max_order+1):
        current_regressors = np.zeros([len(x), order+1])
        current_regressors[:,0] = np.ones(len(x))
        for i in range(1, order+1):
            current_regressors[:,i] = np.power(x, i)
        P.append(current_regressors)

    return P

def EstimateRegressionCoefficients(P_list, y, reg=None):

    # P_list is a list
    # P_list[i] are regressors for order i+1 and is of size N x (order+1), where
    # N is number of data points

    w_list = []
    if reg is None:

        for P in P_list:
            if(P.shape[1] > P.shape[0]): #use dual solution
                w = P.T @ inv(P @ P.T) @ y
            else: # use primal solution
                w = (inv(P.T @ P) @ P.T) @ y
            w_list.append(w)
    else:

       for P in P_list:
           w = (inv(P.T @ P + reg*np.eye(P.shape[1]) ) @ P.T) @ y
           w_list.append(w)

    return w_list


def PerformPrediction(P_list, w_list):

    # P_list is list of regressors
    # w_list is list of coefficients
    # Output is y_predict_mat which N x max_order, where N is the number of samples

    N = P_list[0].shape[0]
    max_order = len(P_list)
    y_predict_mat = np.zeros([N, max_order])
    for order in range(len(w_list)):
        y_predict = np.matmul(P_list[order], w_list[order])
        y_predict_mat[:,order] = y_predict

    return y_predict_mat

# main
# simulation parameters
max_order = 9
reg = 1
np.set_printoptions(precision=4)

# training data
x = np.array([1, 2, 2.3, 2.6, 4.6, 5.0, 5.5, 6.2, 7.2, 7.6])
y = -6 + 5.7*x - 0.6*(x**2) + np.random.random(x.shape[0]);

# test data
xt = np.arange(0, 8, 0.1)
yt = -6 + 5.7*xt - 0.6*(xt**2) + np.random.random(xt.shape[0])*3;

x_plot = np.arange(0, 8, 0.1);

# Create regressors for polynomial order = 9
P_train_list = CreateRegressors(x, max_order)
P_test_list  = CreateRegressors(xt, max_order)
P_plot_list  = CreateRegressors(x_plot, max_order)

# Estimate parameters without regularization
w9 = inv(P_train_list[max_order-1].T@P_train_list[max_order-1])@P_train_list[max_order-1].T@y;

# Estimate parameters without regularization
L = 1
w9_reg = inv(P_train_list[max_order-1].T@P_train_list[max_order-1]+L*np.identity(P_train_list[max_order-1].shape[1]))@P_train_list[max_order-1].T@y;

# Apply prediction

y9_plot = P_plot_list[max_order-1]@w9;
y9_reg_plot = P_plot_list[max_order-1]@w9_reg;

 # Plot and save figures
plt.figure(0, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.scatter(x, y, s=20, marker='o', c='blue', label='train')
plt.scatter(xt, yt, s=20, marker='o', c='red', label='test')

plt.plot(x_plot, y9_plot, linestyle='--', linewidth=1, label='order '+str(max_order))
plt.xlabel('x')
plt.ylabel('y')
plt.title('No Regularization')
plt.plot([0, 8],[0,0], c='black', linewidth=1)
plt.plot([0, 0],[-10,10], c='black', linewidth=1)
plt.xlim(0, 8)
plt.ylim(-10, 10)
plt.legend(loc='lower left',ncol=3, fontsize=15)
plt.savefig('./Lec7_demo1_no_reg.png')


plt.figure(1, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.scatter(x, y, s=20, marker='o', c='blue', label='train')
plt.scatter(xt, yt, s=20, marker='o', c='red', label='test')

plt.plot(x_plot, y9_reg_plot, linestyle='--', linewidth=1, label='order '+str(max_order))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regularization')
plt.plot([0, 8],[0,0], c='black', linewidth=1)
plt.plot([0, 0],[-10,10], c='black', linewidth=1)
plt.xlim(0, 8)
plt.ylim(-10, 10)
plt.legend(loc='lower left',ncol=3, fontsize=15)
plt.savefig('./Lec7_demo1_reg.png')

