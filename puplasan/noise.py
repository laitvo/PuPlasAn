#!/usr/bin/python
# -*- coding: utf-8 -*-


#noise.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020

"""
Signal/noise module
-------------------

Contains functions for estimation of the data noise level.
"""

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from .filters import smooth_ma


def std_ma(noisedata, wpn):
    """Calculates the standard deviation of the difference between *noisedata*
    array and its moving average.

    Parameters
    ----------
    noisedata : 1-D list/array
        array containing noise without signal.
    wpn : int
        Number of the adjacent values of *noisedata* averaged. 
        *wpn* must be an odd number.

    Returns
    -------
    Returns standard deviation of the difference between *noisedata*
    array and its moving average (the standard deviation of the base line 
    corrected data with no signal)."""
    if noisedata.__class__ != np.ndarray:
        noisedata = np.array(noisedata)
    lendata = len(noisedata)
    if lendata < wpn*3:
        warnings.warn('len(noisedata) < wpn*3', UserWarning)
    base = smooth_ma(noisedata, wpn)
    return np.sqrt(np.sum((base - noisedata)**2)/lendata)


def noise_estim_ma(y, wpn, n, m=10):
    """Estimates the *y* data noise level by calculating the noise for *m* 
    short subarrays (of length equal to *n*) randomly selected
    from the *y* array. The resulting noise level corresponds to the minimal
    value of the standard deviations calculated for all short subarrays.

    Parameters
    ----------
    y : 1-D list/array
        array containing both noise as well as signal
    wpn : int
        Number of the adjacent values of *y* averaged. 
        *wpn* must be an odd number.
    n : int
        length of the randomly selected subarray used for calculation
        of the local noise level
    m : int
        number of the randomly selected subarrays (defaul value is 10)

    Returns
    -------
    float
        estimated noise level of the *y* data"""
    stds = []
    mini = 0
    maxi = len(y) - wpn 
    for i1 in range(m):
        rn = np.random.randint(mini, maxi)
        stds.append(std_ma(y[rn:rn+n], wpn=wpn))
    return min(stds)


def noise_estim_ma_xdependent(x, y, wpn, n, partnum, m=10, k=1,
                              fulloutput=False):
    """Estimates the *x*-dependent *y* data noise level.

    Parameters
    ----------
    x : 1-D list/array
        array of independent input data. Must be increasing.
    y : 1-D list/array
        array containing both noise as well as signal, must be of the same
        length as *x*.
    wpn : int
        Number of the adjacent values of *y* averaged. 
        *wpn* must be an odd number.
    n : int
        length of the randomly selected subarray used for calculation
        of the local noise level
    partnum : int
        number of equidistant points for which the *x*-dependent
        noise is calculated
    m : int
        number of the randomly selected subarrays used for calculation
        of the local noise level (defaul value is 10)
    k : int
        Degree of the smoothing spline. Must be <= 5. Default is k=1,
        a linear spline.
    fulloutput : bool
        Influences the output (see the *Returns* section).
        Default value is *False*.

    Returns
    -------
    If *fulloutput* is *True*, returns:
    numpy.array
        array of the same length as *x* containing the *x*-dependent
        *y* data noise level
    instance of the *scipy.interpolate.fitpack2.LSQUnivariateSpline* class
    list
        x-values (positions) of the calculated noise
        (list of length equal to *partnum*)
    list
        list of *x*-dependent *y* data noise corrresponding
        to the previous output

    Else returns a numpy.array of the same length as *x* containing
    the *x*-dependent *y* data noise."""
    partlen = int(len(x)/partnum)
    if partlen <= 1:
        raise Exception('"partnum" is too high ' +
                        'or length of "x" and "y" is to low')
    sdevx = []
    sdevy = []
    for i1 in range(partnum):
        ind1 = i1*partlen
        ind2 = i1*partlen + partlen
        sdevx.append(sum(x[ind1:ind2])/(ind2-ind1))
        sdevy.append(noise_estim_ma(y[ind1:ind2], wpn=wpn, n=n, m=m))
    sdevx = [x[0]] + sdevx
    sdevy = [sdevy[0]] + sdevy
    sdevx.append(x[-1])
    sdevy.append(sdevy[-1])
    spl = UnivariateSpline(sdevx, sdevy, k=k, s=0)
    if fulloutput:
        return spl(x), spl, sdevx, sdevy
    else:
        return spl(x)


