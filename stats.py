import numpy as np
from scipy.special import erfcinv, gammaincc, gamma
from scipy.optimize import fsolve

def get_sigma_from_chisq(chisq, dof):
    arg = chisq + np.log((2/chisq)**(dof-2) * gamma(dof/2)**2)
    if chisq > 20:
        return np.sqrt(np.log(2 / np.pi) + arg - np.log(arg))
    return np.sqrt(2) * erfcinv(gammaincc(dof/2, chisq/2))

def get_threshold(image, percentile):
    # Returns a number such that (percentile)% of the image values are above that number
    width = np.max(image) / 100
    def theta_func(image, thresh):
        # 1 if image > thresh, 0 otherwise, blur in between
        return (np.tanh((image - thresh) / width) + 1) / 2
    def solve_func(thresh):
        return np.sum(image * theta_func(image, thresh)) / np.sum(image) * 100 - percentile
    output = fsolve(solve_func, np.mean(image))
    return output[0]