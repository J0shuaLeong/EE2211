
# EE2211 Lecture 5 Demo 1 linear regression
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
## 
X = np.array([[1, -9], [1, -7], [1, -5], [1, 1], [1, 5], [1, 9]])
Y = np.array([[-6], [-6], [-4], [-1], [1], [4]])
w = inv(X.T @ X) @ X.T @ Y
print(w)

Xnew = np.array([1, -1])
Ynew = Xnew@w
print(Ynew)

## difference
print("Mean squared error between Y and XW")
Ytest=X@w
MSE = np.square(np.subtract(Ytest,Y)).mean()
print(MSE)
MSE = mean_squared_error(Ytest,Y)
print(MSE)


# EE2211 Lecture 5 Demo 2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
## 
X = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 3], [1, 1, 0]])
Y = np.array([[1, 0], [0, 1], [2, -1], [-1, 3]])
w = inv(X.T @ X) @ X.T @ Y
print("the estimated w")
print(w)

Xnew = np.array([[1, 6, 8], [1, 0, -1]])
Ynew = Xnew@w
print("testing Ynew")
print(Ynew)

## difference
print("Mean squared error between Y and XW")
Ytest=X@w
MSE = np.square(np.subtract(Ytest,Y)).mean()
print(MSE)
MSE = mean_squared_error(Ytest,Y)
print(MSE)

# EE2211 Lecture 5 Demo 3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
## 
X = np.array([[1, 3], [1, 4], [1, 10], [1, 6], [1, 7]])
Y = np.array([[0, 5], [1.5, 4], [-3, 8], [-4, 10], [1, 6]])
w = inv(X.T @ X) @ X.T @ Y
print('W')
print(w)

Xnew = np.array([[1, 2]])

Ynew = Xnew@w

print('Ynew')
print(Xnew)
print(Xnew.shape)
print(Ynew)

Ytest= X@w
print('Ytest')
print(Ytest)
plt.plot(X[:,1], Y[:,0], 'o', label = 'Y1')
plt.plot(X[:,1], Y[:,1], 'x', label = 'Y2')
plt.plot(X[:,1], Ytest[:,0])
plt.plot(X[:,1], Ytest[:,1])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

