import numpy as np
from scipy.signal import convolve
from scipy.optimize import fsolve

def blur(image, sigma):
    # Blur the image where sigma is in units of pixels
    if sigma == 0: return image
    width = int(np.ceil(3 * sigma))
    line = np.arange(-width, width+1)
    xs, ys = np.meshgrid(line, line)
    gauss = np.exp(-(xs**2 + ys**2) / (2*sigma**2))
    gauss /= np.sum(gauss)
    flat = convolve(np.ones_like(image), gauss, mode="same")
    return convolve(image, gauss, mode="same") / flat

def weighted_blur(image, weights, sigma):
    # Blur the image where sigma is in units of pixels
    return blur(image*weights, sigma) / blur(weights, sigma)

def blur1d(image, sigma):
    # Blur the image where sigma is in units of pixels
    width = int(np.ceil(3 * sigma))
    line = np.arange(-width, width+1)
    gauss = np.exp(-line**2 / (2*sigma**2))
    gauss /= np.sum(gauss)
    flat = convolve(np.ones_like(image), gauss, mode="same")
    return convolve(image, gauss, mode="same") / flat

def get_threshold(image, percentile, tolerance=1e-3, maxiter=1000):
    """
    Returns a number such that (percentile)% of the image values are above that number
    """
    quantile = percentile/100
    high = np.max(image)    
    low = np.min(image) 
    
    observed_quantile = np.inf
    n_iter = 0
    while np.abs(observed_quantile - quantile) > tolerance:
        mid = (high + low) / 2
        observed_quantile = np.sum(image[image > mid]) / np.sum(image)
        if observed_quantile > quantile:
            # Raise the threshold
            low = mid
        else:
            # Lower the threshold
            high = mid

        n_iter += 1
        if n_iter > maxiter:
            print("Warning: maximum number of iterations exceeded in get_threshold")
            break
    return mid