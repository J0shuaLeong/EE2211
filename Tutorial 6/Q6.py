import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target

encoder = OneHotEncoder()
y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.26, random_state = 0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis = 1)
y_true_labels = np.argmax(y_test, axis = 1)
classification = accuracy_score(y_true_labels, y_pred_labels, normalize = False)
print("Classified correctly: ")
print(classification)

# Polynomial Regression
poly = PolynomialFeatures(2)
P = poly.fit_transform(X_train)
model.fit(P, y_train)
X_test_poly = poly.transform(X_test)
y_pred_poly = model.predict(X_test_poly)
y_pred_poly_label = np.argmax(y_pred_poly, axis=1)
classification_poly = accuracy_score(y_true_labels, y_pred_poly_label, normalize = False)
print("Poly Classification")
print(classification_poly)
