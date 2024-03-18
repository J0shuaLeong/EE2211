import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0254586R(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    pass
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    InvXTX = np.linalg.inv(XTX)
    w = np.dot(np.dot(InvXTX, XT), y)
    
    # return in this order
    return InvXTX, w


