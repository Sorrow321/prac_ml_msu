import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    f_w = function(w)

    res = np.zeros(len(w), dtype=np.float64)
    for i in range(len(w)):
        w[i] += eps
        res[i] = function(w)
        w[i] -= eps

    return (res - f_w) / eps