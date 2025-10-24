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
    stretch = np.cos(ycol.coord_ref_value * np.pi / 180)
    xs = (x - xcol.coord_ref_point)/stretch * xcol.coord_inc + xcol.coord_ref_value
    ys = (y - ycol.coord_ref_point) * ycol.coord_inc + ycol.coord_ref_value
    return xs, ys

def from_hms(s):
    # Convert the passed string from hms to degrees
    h, m, s = s.split(':')
    return (float(h) + float(m) / 60 + float(s)/3600) * 360 / 24

def to_hms(d, arcsec_precision=None):
    # Convert the passed string from dms to degrees
    x = d * 24 / 360
    h = int(x)
    x = (x- int(x)) * 60
    m = int(x)
    x = (x- int(x)) * 60
    s = x
    if arcsec_precision is not None:
        s = round(s, arcsec_precision)
    return f"{h:02d}:{m:02d}:{s}"

def from_dms(s):
    # Convert the passed string to hms from degrees
    d, m, s = s.split(':')
    x = np.abs(float(d)) + float(m) / 60 + float(s)/3600
    return np.sign(float(d)) * x

def to_dms(d, arcsec_precision=None):
    # Convert the passed string to dms from degrees
    sign = "-" if d < 0 else ""
    x = np.abs(d)
    d = int(x)
    x = (x- int(x)) * 60
    m = int(x)
    x = (x- int(x)) * 60
    s = x
    if arcsec_precision is not None:
        s = round(s, arcsec_precision)
    return f"{sign}{d:02d}:{m:02d}:{s}"