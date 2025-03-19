import numpy as np

def sky_to_ra_dec(x, y, xcol, ycol):
    xs = (x - xcol.coord_ref_point) * xcol.coord_inc * np.pi / 180
    ys = (y - ycol.coord_ref_point) * ycol.coord_inc * np.pi / 180
    phi = np.arctan2(-xs, ys)
    theta = np.arctan2(1, np.sqrt(xs**2 + ys**2))
    dec_0 = ycol.coord_ref_value * np.pi / 180
    ra = xcol.coord_ref_value + 180 / np.pi * np.arctan2(
        -np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.cos(dec_0) - np.cos(theta) * np.sin(dec_0) * np.cos(phi)
    )
    dec = 180 / np.pi * np.arcsin(
        np.sin(theta) * np.sin(dec_0)
        + np.cos(theta) * np.cos(dec_0) * np.cos(phi)
    )
    return ra, dec


def rough_sky_to_ra_dec(x, y, xcol, ycol):
    xs = (x - xcol.coord_ref_point) * xcol.coord_inc + xcol.coord_ref_value
    ys = (y - ycol.coord_ref_point) * ycol.coord_inc + ycol.coord_ref_value
    return xs, ys