## (a) split data 
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris_dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.26, random_state=0)


## (b) one-hot encoding
# Ytr_onehot = list()
# for i in y_train:
# letter = [0, 0, 0]
# letter[i] = 1
# Ytr_onehot.append(letter)
# Yts_onehot = list()
# for i in y_test:
# letter = [0, 0, 0]
# letter[i] = 1
# Yts_onehot.append(letter)
from sklearn.preprocessing import OneHotEncoder
onehot_encoder=OneHotEncoder(sparse=False)
reshaped = y_train.reshape(len(y_train), 1)
Ytr_onehot = onehot_encoder.fit_transform(reshaped)
reshaped = y_test.reshape(len(y_test), 1)
Yts_onehot = onehot_encoder.fit_transform(reshaped)


## (c) Linear Classification
from numpy.linalg import inv
import numpy as np
w = inv(X_train.T @ X_train) @ X_train.T @ Ytr_onehot
print("Weight:")
print(w)
# calculate the output based on the estimated w and test input X and then assign to one of the classes based on one hot encoding
yt_est = X_test.dot(w); 
yt_cls = [[1 if y == max(x) else 0 for y in x] for x in yt_est]
print("Output:")
print(yt_cls)
# compare the predicted y with the ground truth
m1 = np.matrix(Yts_onehot)
m2 = np.matrix(yt_cls)
difference = np.abs(m1 - m2)
print("Difference:")
print(difference)
# calculate the error rate/accuracy
correct = np.where(~difference.any(axis=1))[0]
accuracy = len(correct)/len(difference)
print("Number of correct predictions: " + str(len(correct)))
print("Accuracy: " + str(accuracy) + "\n\n")


## (d) Polynomial Classification
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
P = poly.fit_transform(X_train)
Pt = poly.fit_transform(X_test)
if P.shape[0] > P.shape[1]:
    wp = inv(P.T @ P) @ P.T @ Ytr_onehot
else:
    wp = P.T @ inv(P @ P.T) @ Ytr_onehot
print("Weight:")
print(wp)
yt_est_p = Pt.dot(wp); 
yt_cls_p = [[1 if y == max(x) else 0 for y in x] for x in yt_est_p]
print("Output:")
print(yt_cls_p)
m1 = np.matrix(Yts_onehot)
m2 = np.matrix(yt_cls_p)
difference = np.abs(m1 - m2)
print("Difference:")
print(difference)
correct_p = np.where(~difference.any(axis=1))[0]
accuracy_p = len(correct_p)/len(difference)
print("Number of correct predictions: " + str(len(correct_p)))
print("Accuracy: " + str(accuracy_p))
