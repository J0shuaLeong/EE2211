
# EE2211 Lecture 4 Transpose and rank Demo 1

import numpy as np
from numpy.linalg import matrix_rank
#X = np.array([[1, 4, 3], [0, 4, 2], [3, -2, 9]])
X = np.array([[1, 4, 3], [0, 4, 2], [1, 8, 5]])
print(matrix_rank(X))
print(X)
print(X.T)




# EE2211 Lecture 4 Product and inverse Demo 2
import numpy as np
from numpy.linalg import inv
X = np.array([[1, 4], [0, 4], [3, -2]]) # size 3 x 2
y = np.array([3, 0.5, 4]) # size 3
print(y)
print(y.shape)

y1 = np.array([[3], [0.5], [4]]) # size 3 x 1
print(y1)
print(y1.shape)

z = X.T@y # 2 x 3 times 3 = 2
print('Vector-matrix product')
print(z)

z1 = X.T@y1 # 2 x 3 times 3 x 1 = 2 x 1
print('Vector-matrix product')
print(z1)

print('matrix product')
X2 = np.array([[1, 4, 3], [0, 4, 2]]) # size 2 x 3
Q = X@X2 # 3 x 2 times 2 x 3 --> 3 x 3
print(Q)

print('matrix inverse')
X = np.array([[1, 4, 3], [0, 4, 2], [3, -2, 9]])
print(inv(X))



# EE2211 Lecture 4 Even-determined system (m = d) Demo 3


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

X = np.array([[1, 1], [1, -2]])
y = np.array([4, 1])
print(y.shape)
w = inv(X) @ y
print(w)

print('more examples')
X = np.array([[1, 4, 2], [0, 4, 3], [3, 4, 9]])
y = np.array([39, 40, 50])
w = inv(X) @ y
print(w)

print('rank')
X = np.array([[1, 4, 2], [0, 4, 3], [1, 8, 5]])
y = np.array([1, 0, 1])
print(matrix_rank(X))



# EE2211 Lecture 4 Over-determined system (m > d) Demo 4

import numpy as np
from numpy.linalg import inv
X = np.array([[1, 1], [1, -1], [1, 0]])
y = np.array([1, 0, 2])
w = inv(X.T @ X) @ X.T @ y
print(w)




# EE2211 Lecture 4 Under-determined system 2 (m < d) Example 3

import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_rank
from numpy.linalg import det

X = np.array([[1, 2, 3], [1, -2, 3]])
y = np.array([2, 1])
w = X.T @ inv(X@ X.T) @ y
print(w)



