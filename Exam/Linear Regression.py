import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def left_inverse(X, y):
    invXTX = inv(X.T @ X)
    w = invXTX @ X.T @ y
    print("")
    print("left inverse")
    print("")
    return w
    
def right_inverse(X, y):
    invXXT = inv(X @ X.T)
    w = X.T @ invXXT @ y
    print("")
    print("right inverse")
    print("")
    return w
    
def even_system(X, y):
    print("")
    print("even")
    print("")
    return inv(X) @ y

# Function to take user input for matrix
def get_matrix_input(rows, columns):
    matrix = []
    print(f"Enter the elements of the matrix ({rows}x{columns}):")
    for i in range(rows):
        row = list(map(float, input().split()))
        matrix.append(row)
    return np.array(matrix)

def add_bias(X):
    ones_column = np.ones((X.shape[0], 1))
    X_with_bias = np.concatenate((ones_column, X), axis=1)
    return X_with_bias

def mean_square(X, Y, w):
    print("")
    print("Mean squared error between Y and XW")
    Ytest=X@w
#     MSE = np.square(np.subtract(Ytest,Y)).mean()
#     print(MSE)
    MSE = mean_squared_error(Ytest,Y)
    print(MSE)
    
def mse(y):
    total = 0;
    for x in y:
        total += x
    mY = total / len(y)
    errorSquare = 0
    for x in y:
        error = x - mY
        errorSquare += error ** 2
    print("Yuyang mse: ")
    print(errorSquare / len(y))
    

# def plot_results(X, y, w):
#     plt.scatter(X[:, 1], y, color='blue', label='Training data')

#     fitted_line_x = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
#     fitted_line_y = np.dot(add_bias(np.expand_dims(fitted_line_x, axis=1)), w)

#     plt.plot(fitted_line_x, fitted_line_y, color='red', label='Fitted line')
#     plt.title('Linear Regression')
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.legend()
#     plt.show()

# Main function
def main():
    
    # Get user input for training set X
    print("Enter the dimensions of matrix X (rows x columns):")
    rows_X = int(input("Rows: "))
    columns_X = int(input("Columns: "))
    X = get_matrix_input(rows_X, columns_X)

    # Get user input for target vector y
    print("")
    print("Enter the dimensions of vector y (rows x 1):")
    rows_y = int(input("Rows: "))
    columns_y = int(input("Columns: "))
    y = get_matrix_input(rows_y, columns_y)
    
    print("")
    add_bias_option = input("Do you want to add bias to X? (y/n): ").lower()
    if add_bias_option == "y":
        columns_X += 1
        X = add_bias(X)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0, test_size=0.26, random_state=0)
    
    if rows_X > columns_X:
        w = left_inverse(X, y)
    elif rows_X < columns_X:
        w = right_inverse(X, y)
    else:
        w = even_system(X, y)

    # Compute the least squares solution
    print("")
    print("Least squares solution (w):")
    print(w)
    print("")
    
    find_mean_square = input("Find MSE (y/n)?").lower()
    if find_mean_square == "y":
        mean_square(X, y, w)
        mse(y)
    print("")

    find_new_y = input("Predict y with new x (y/n)?").lower()
    while find_new_y == "y":
        # Take user input for new x
        print("Enter the new x vector (rows x columns):")
        rows_X = int(input("Rows: "))
        columns_X = int(input("Columns: "))
        new_x = get_matrix_input(rows_X, columns_X)

        if add_bias_option == "y":
            new_x = add_bias(new_x)

        print("")

        # Calculate the new y using the calculated w
        new_y = new_x @ w
        print("Corresponding new y:")
        print(new_y)
        print("")
        
        find_new_y = input("Predict y with new x (y/n)?").lower()

    # plot_results_option = input("Plot the result, w (y/n)?").lower()
    # if plot_results_option == "y":
    #     plot_results(X, y, w)

if __name__ == "__main__":
    main()