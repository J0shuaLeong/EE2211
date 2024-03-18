import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
y = np.array([[5],[5],[4],[3],[2],[2]])

order = 3
poly = PolynomialFeatures(order)
P = poly.fit_transform(X)

model = LinearRegression()
model.fit(P, y)

# Generate points for the fitted curve
x_fit = np.linspace(min(X), max(X), 100).reshape(-1, 1)
P_fit = poly.transform(x_fit)
y_fit = model.predict(P_fit)

X_test = np.array([[9]])
P_test = poly.transform(X_test)
y_pred = model.predict(P_test)

print("polynomial Regression: ")
print(y_pred[0])

# Linear Regression
model = LinearRegression()
model.fit(X, y)
test_prediction = model.predict(X_test)
print("Linear regression: ")
print(test_prediction)

plt.scatter(X, y, color = 'blue')
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve (3rd Order Polynomial)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Fitting with 3rd Order Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()