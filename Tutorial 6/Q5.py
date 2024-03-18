import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[-1], [0], [0.5], [0.3], [0.8]])
y = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

model = LinearRegression()
model.fit(X, y)

X_test = np.array([[-0.1], [0.4]])
y_pred = model.predict(X_test)
predicted_classes = np.argmax(y_pred, axis=1) + 1
print("Using Linear Regression")
print(predicted_classes[0])
print(predicted_classes[1])

order = 5
poly = PolynomialFeatures(order)
P = poly.fit_transform(X)
model.fit(P, y)
X_test_poly = poly.transform(X_test)
y_pred_poly = model.predict(X_test_poly)
predicted_classes_poly = np.argmax(y_pred, axis=1) + 1
print("Using Poly")
print(predicted_classes_poly[0])
print(predicted_classes_poly[1])