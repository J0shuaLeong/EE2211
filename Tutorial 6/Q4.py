import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, -1], [1, 0], [1, 0.5], [1, 0.3], [1, 0.8]])
y = np.array([1, 1, -1, 1, -1])

w = inv(X.T @ X) @ X.T @ y
print(w)
XT = np.array([[1, -0.1], [1, 0.4]])
y_predict = XT @ w
print(y_predict)
y_class_predict = np.sign(y_predict)
print(y_class_predict)