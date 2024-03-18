import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


X_train = np.array([[1, -10], [1, -8], [1, -3], [1, -1], [1, 2], [1, 7]])
y_train = np.array([[4.18], [2.42], [0.22], [0.12], [0.25], [3.09]])

X_test = np.array([[1, -9], [1, -7], [1, -5], [1, -4], [1, -2], [1, 1], [1, 4], [1, 5], [1, 6], [1, 9]])
y_test = np.array([[3], [1.81], [0.8], [0.25], [-0.19], [0.4], [1.24], [1.68], [2.32], [5.05]])

def noRegularisation():
    train_mses = []
    test_mses = []
    degrees = range(1, 7)

    for i in degrees:
        poly_features = PolynomialFeatures(degree=i)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.fit_transform(X_test)

        model = LinearRegression()
        model.fit(X_poly_train, y_train)

        train_pred = model.predict(X_poly_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mses.append(train_mse)

        test_pred = model.predict(X_poly_test)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mses.append(test_mse)

    print("part a: ")
    for i, train_mse, test_mse in zip(range(1, 7), train_mses, test_mses):
        print(f'Degree {i}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}')

    print("")

    plt.plot(degrees, train_mses, marker='o', label='Training MSE')
    plt.plot(degrees, test_mses, marker='o', label='Test MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE for Different Polynomial Orders')
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.show()

def withRegularisation():
    train_mses = []
    test_mses = []
    degrees = range(1, 7)

    for i in degrees:
        poly_features = PolynomialFeatures(degree=i)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.fit_transform(X_test)

        model = Ridge(alpha=1)
        model.fit(X_poly_train, y_train)

        train_pred = model.predict(X_poly_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mses.append(train_mse)

        test_pred = model.predict(X_poly_test)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mses.append(test_mse)

    print("part b: ")
    for i, train_mse, test_mse in zip(range(1, 7), train_mses, test_mses):
        print(f'Degree {i}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}')

    print("")

    plt.plot(degrees, train_mses, marker='o', label='Training MSE')
    plt.plot(degrees, test_mses, marker='o', label='Test MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE for Different Polynomial Orders with ridge')
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.show()

noRegularisation()
withRegularisation()