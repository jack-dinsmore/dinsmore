import numpy as np
from scipy.special import erfcinv, gammaincc, gamma

def get_sigma_from_chisq(chisq, dof, return_p=False):
    arg = chisq + np.log((2/chisq)**(dof-2) * gamma(dof/2)**2)
    if chisq > 20:
        return np.sqrt(np.log(2 / np.pi) + arg - np.log(arg))
    p = gammaincc(dof/2, chisq/2)
    z = get_sigma_from_p(p)
    if return_p:
        return z, p
    else:
        return z

def get_sigma_from_p(p):
    return np.sqrt(2) * erfcinv(p)
    
def draw_from_image(image, linex, liney, n):
    cdf = np.cumsum(image.reshape(-1))
    cdf /= cdf[-1]
    index = np.floor(np.interp(np.random.random(n), cdf, np.arange(len(cdf)))).astype(int)
    x = linex[index // image.shape[0]]
    y = liney[index % image.shape[0]]
    x += np.random.random(n) * (linex[1] - linex[0])
    y += np.random.random(n) * (liney[1] - liney[0])
    return x, y