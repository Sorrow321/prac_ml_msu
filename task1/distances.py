import numpy as np


# found this brief and efficient code in this article
# https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
def euclidean_distance(X, Y):
    return np.sqrt(-2 * np.dot(X, Y.T) + np.sum(X ** 2, axis=1)[:, np.newaxis]\
    + np.sum(Y ** 2, axis = 1).T)

def cosine_distance(X, Y):
    return 1 - np.dot(X, Y.T) / np.dot(np.linalg.norm(X, axis=1)[:, None],\
    np.linalg.norm(Y, axis=1)[:, None].T)
