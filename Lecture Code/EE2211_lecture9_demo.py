
# Lecture 9 (demo - decision tree & housing price)

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def our_own_tree(y):

    # split data at first level
    # L stands for left, R stands for right
    yL, yR = find_best_split(y)

    # compute prediction
    yL_pred = np.mean(yL)*np.ones(len(yL))
    yR_pred = np.mean(yR)*np.ones(len(yR))

    y_pred = np.concatenate([yL_pred, yR_pred])

    return y_pred

#Go through all possible thresholds to determine the best split based on MSE
def find_best_split(y):

    # index represents last element in the below threshold node
    sq_err_vec = np.zeros(len(y)-1)
    meansq_err_vec = np.zeros(len(y)-1)
    for index in range(0, len(y)-1):

        # split the data
        data_below_threshold = y[:index+1]
        data_above_threshold = y[index+1:]

        # Compute estimate
        mean_below_threshold = np.mean(data_below_threshold)
        mean_above_threshold = np.mean(data_above_threshold)

        # Compute total square error
        # Note that MSE = total square error divided by number of data points
        below_sq_err = np.sum(np.square(data_below_threshold - mean_below_threshold))
        above_sq_err = np.sum(np.square(data_above_threshold - mean_above_threshold))
        sq_err_vec[index] = below_sq_err + above_sq_err
        meansq_err_vec[index] = sq_err_vec[index]/len(y)

    #print out MSE
    print('MSE list for house size')
    print(meansq_err_vec)
    best_index = np.argmin(meansq_err_vec)
    yL = y[:best_index+1]
    yR = y[best_index+1:]
    print('Minimum MSE = '+str(meansq_err_vec[best_index])+' at threshold index '+str(best_index+1))
    return yL, yR


#main
#Input (house size and rooms) and output (price)
S = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 3.2, 3.8])
R = np.array([2,1,3,5,4,6,7])
P = np.array([0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.80])

#sort
sort_index = S.argsort()
S = S[sort_index]
R = R[sort_index]
P = P[sort_index]
print(S)

# scikit decision tree regressor
scikit_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=1)
# Focus on House Size
scikit_tree.fit(S.reshape(-1,1), P) # reshape necessary because tree expects 2D array
scikit_tree_predict = scikit_tree.predict(S.reshape(-1,1))

# Our own tree regressor
tree_predict = our_own_tree(P)

# Plot
plt.figure(0, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.scatter(S, P, c='steelblue', s=30)
plt.plot(S, scikit_tree_predict, color='black', lw=2, label='scikit-learn')
plt.plot(S, tree_predict, color='red', linestyle='--', lw=2, label='our own tree')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.legend(loc='upper right',ncol=3, fontsize=10)
plt.savefig('Fig_Lec9_decisiontree.png')

