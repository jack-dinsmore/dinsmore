import numpy as np
from scipy.signal import convolve

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

def extract_square(image, center, halfwidth, fill_value=np.nan):
    y0, x0 = center
    size = 2 * halfwidth + 1

    cutout = np.zeros((size, size), dtype=image.dtype)+fill_value

    # Desired image bounds
    y1_img = max(0, y0 - halfwidth)
    y2_img = min(image.shape[0], y0 + halfwidth + 1)
    x1_img = max(0, x0 - halfwidth)
    x2_img = min(image.shape[1], x0 + halfwidth + 1)

    # Corresponding cutout bounds
    y1_cut = y1_img - (y0 - halfwidth)
    y2_cut = y1_cut + (y2_img - y1_img)
    x1_cut = x1_img - (x0 - halfwidth)
    x2_cut = x1_cut + (x2_img - x1_img)

    cutout[y1_cut:y2_cut, x1_cut:x2_cut] = image[y1_img:y2_img, x1_img:x2_img]

    return cutout

def shift_image(img, dx=0, dy=0, fill=0):
    """
    Shift a 2D (or higher, with last two axes as y,x) image by integer pixels.

    Parameters
    ----------
    img : np.ndarray
    dx : int
        Shift in x (columns). Positive = right.
    dy : int
        Shift in y (rows). Positive = down.
    fill : scalar, optional
        Fill value for emptied pixels (default 0).

    Returns
    -------
    np.ndarray
        Shifted image (same shape as input).
    """
    out = np.full_like(img, fill)

    # Source and destination slices
    y_src_start = max(0, -dy)
    y_src_end   = img.shape[-2] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end   = img.shape[-1] - max(0, dx)

    y_dst_start = max(0, dy)
    y_dst_end   = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx)
    x_dst_end   = x_dst_start + (x_src_end - x_src_start)

    out[..., y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
        img[..., y_src_start:y_src_end, x_src_start:x_src_end]

    return out