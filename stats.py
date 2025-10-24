import numpy as np
from scipy.special import erfcinv, gammaincc, gamma

def get_sigma_from_chisq(chisq, dof):
    arg = chisq + np.log((2/chisq)**(dof-2) * gamma(dof/2)**2)
    if chisq > 20:
        return np.sqrt(np.log(2 / np.pi) + arg - np.log(arg))
    return np.sqrt(2) * erfcinv(gammaincc(dof/2, chisq/2))