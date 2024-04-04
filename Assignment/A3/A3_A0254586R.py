import numpy as np
import math

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0254586R(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    a = 2.5
    b = 0.6
    c = 2
    d = 3

    a_out = []
    b_out = []
    c_out = []
    d_out = []
    f1_out = []
    f2_out = []
    f3_out = []

    for iteration in range(num_iters):
        grad_a = 4 * a**3
        grad_b = 2 * math.sin(b) * math.cos(b)
        grad_c = 5 * c**4
        grad_d = 2 * d * (math.sin(d)) + (d**2) * (math.cos(d))
        a = a - learning_rate * grad_a
        b = b - learning_rate * grad_b
        c = c - learning_rate * grad_c
        d = d - learning_rate * grad_d
        a_out.append(a)
        b_out.append(b)
        c_out.append(c)
        d_out.append(d)
        cost_a = a**4
        cost_b = (math.sin(b)) * (math.sin(b))
        cost_c = c**5
        cost_d = (d**2) * (math.sin(d))
        f1_out.append(cost_a)
        f2_out.append(cost_b)
        f3_out.append(cost_c + cost_d)

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 