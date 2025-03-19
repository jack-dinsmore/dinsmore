import numpy as np
from scipy.signal import convolve

def blur(image, sigma):
    # Blur the image where sigma is in units of pixels
    width = int(np.ceil(3 * sigma))
    line = np.arange(-width, width+1)
    xs, ys = np.meshgrid(line, line)
    gauss = np.exp(-(xs**2 + ys**2) / (2*sigma**2))
    gauss /= np.sum(gauss)
    return convolve(image, gauss, mode="same")