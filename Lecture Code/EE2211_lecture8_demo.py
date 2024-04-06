# Lecture 8 Demo 1 - Gradient descent


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Main function is y=x**2
# Gradient is 2x
# Initialization & Parameters
x = -1
eta = 0.4 #just nice
print('------ eta = '+str(eta)+' -------')
for i in range(0,5):
 x= x-2*eta*x
 print(x)


x = -1
eta = 0.01 #too small
print('------ eta = '+str(eta)+' -------')
for i in range(0,5):
 x= x-2*eta*x
 print(x)


x = -1
eta = 0.9 #too big
print('------ eta = '+str(eta)+' -------')
for i in range(0,5):
 x= x-2*eta*x
 print(x)