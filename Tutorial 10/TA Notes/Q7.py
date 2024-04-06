import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(Train=False):
    import csv
    data = []
    ## Read the training data
    f = open('./spambase/spambase.data')
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        data.append(row)
    f.close()
    ## x[:-1]: omit the last element of each x row
    X = np.array([x[:-1] for x in data]).astype(float)
    ## x[-1]: the first element from the right instead of from the left 
    y = np.array([x[-1] for x in data]).astype(float)
    del data # free up the memory
    if Train:
        # returns X_train, X_test, y_train, y_test
        return train_test_split(X, y, test_size=0.2, random_state=8)
    else:
        return X, y
    
## Get training and test sets 
X_train, X_test, y_train, y_test = load_data(Train=True)
## Linear regression
inv_XTX = np.linalg.inv(X_train.transpose().dot(X_train))
pinv = inv_XTX.dot(X_train.transpose())
W = pinv.dot(y_train)
## Prediction
y_predict = X_test.dot(W)
## Calculate classification error rate
yp_cls = [1 if yout >= 0.5 else 0 for yout in y_predict] 
difference = np.abs(y_test - yp_cls)
test_error_count = (difference == 1).sum()
test_error_rate = test_error_count/len(y_test)
print(test_error_rate)
##--- Compute FPR and FNR at different thresholds ---##
## separate the two classes of predicted data based on the ground truth y_test
pos_idx = np.where(y_test == 1) # identify the indexing of positive-class in the test set
neg_idx = np.where(y_test == 0) # identify the indexing of negative-class in the test set
y_predict_for_PosSamples = y_predict[pos_idx] # prediction of the positive-class data
y_predict_for_NegSamples = y_predict[neg_idx] # prediction of the negative-class data
## use the shorter among the two arrays as threshold
if ( len(y_predict_for_PosSamples) <= len(y_predict_for_NegSamples) ): 
    sorted = np.sort(y_predict_for_PosSamples) # sort in ascending order to be used as threshold
else:
    sorted = np.sort(y_predict_for_NegSamples) # sort in ascending order to be used as threshold
FNR = []
FPR = [] 
TPR = [] 
## Compute FNR, FPR, and TPR for each threshold
for k in range(len(sorted)):
    yp_cls_pos = np.abs([1 if yout >= sorted[k] else 0 for yout in y_predict_for_PosSamples]) 
    yp_cls_neg = np.abs([1 if yout >= sorted[k] else 0 for yout in y_predict_for_NegSamples]) 
    FNR += [(yp_cls_pos == 0).sum()/len(y_predict_for_PosSamples)]
    FPR += [(yp_cls_neg == 1).sum()/len(y_predict_for_NegSamples)]
    TPR += [1-(yp_cls_pos == 0).sum()/len(y_predict_for_PosSamples)]
##--- Plot ROC and DET curves ---##
plt.plot(FPR, FNR, '-', label = 'DET') # Detection error tradeoff
plt.plot(FPR, TPR, '-', label = 'ROC') # Receiver operating characteristic
plt.xlabel('FPR') 
plt.ylabel('FNR/TPR')
plt.legend(fontsize=15)
plt.show()
##--- Comput AUC ---##
ypos_array = [[1 if y_predict_for_PosSamples[j] >= y_predict_for_NegSamples[k] else 0 for j in 
range(len(y_predict_for_PosSamples))] for k in range(len(y_predict_for_NegSamples))] 
AUC = np.sum(ypos_array)/(len(y_predict_for_PosSamples)*len(y_predict_for_NegSamples))
print(AUC)