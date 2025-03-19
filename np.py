import numpy as np

def nd_argmin(array):
    '''
    Returns an array containing the coordinates of the smallest point in the array
    '''
    flat_argmin = np.argmin(array.reshape(-1))
    shape_dim = np.cumprod(np.flip(array.shape))
    shape_dim = np.concatenate([[1],shape_dim[:-1]])
    out = []
    for s in np.flip(shape_dim):
        index = flat_argmin // s
        out.append(index)
        flat_argmin -= s * index
    return np.array(out)