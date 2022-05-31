#!/usr/bin/python
# -*- coding: utf-8 -*-


#bline.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020

'''
Baseline correction module
--------------------------

This module contains functions for determination the base line
of experimental data (e.g. UV/VIS spectrum). The base line is obtained
by iterative removing the high frequency signal and interpolating
the remaining points by the spline method.
'''

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from .filters import smooth_ma


def base(x, y, wpn, noise, rem=0.5, k=1, fulloutput=False, cbfun=None,
         cbfunargs=(), remrate=1.5):
    """Calculates the baseline of the input data *y* using an iterative removing
    of the high frequency signal. The method is based on application of moving
    average smoothing.

    Parameters
    ----------
    x : 1-D list/array
        array of independent input data. Must be increasing.
    y : 1-D list/array
        array of dependent input data, of the same length as *x*.
    wpn : int
        Number of the adjacent values of *y* averaged. *wpn* must be an odd
        number.
    noise : float or 1-D numpy.array
        Estimated noise level of the data in the *y* array. If *noise*
        is an array, it must be of the same length as x (x-dependent noise).
    rem : float
        During the calculation the elements of *y* containing the high
        frequency signal are removed iteratively. The iterations are stopped
        when the number of the remining elements in the *y* is equal or lower
        then lenx*(1. - rem), where lenx is the original length
        of the *y* array.
    k : int
        Degree of the smoothing spline. Must be <= 5. Default is k=1,
        a linear spline.
    fulloutput : bool
        Influences the output (see the *Returns* section).
        Default value is *False*.
    cbfun : callable/None, optional *cbfun(b_inds, ma, \*cbfunargs)*
        A callback function executed at the end of each iteration
        (*cbfun(b_inds, ma, *cbfunargs)*),
        where *b_inds* is an numpy.array containing the actual indexes
        of the array *y* which are used to construct the baseline,
        *ma* is an array of smoothed values of *y* corresponding to the indexes
        in the *b_inds* array
    cbfunargs : tuple, optional
        Extra positional arguments passed to *cbfun*
    remrate : float
        A number in (1,inf> range (reasonable values are
        between 1.05 and 3). The higher is the value the more signal points
        are removed during one iteratoion (default value is equal to 1.5)

    Returns
    -------
    If *fulloutput* is *True*, returns a numpy.array of the same length as *x*
    containing the *y* data base line, the instance of the *scipy.interpolate.
    fitpack2.LSQUnivariateSpline* class and an numpy.array of the indexes
    of *y* corresponding to the base line with the high frequency
    signal removed.

    Else returns a numpy.array of the same length as *x* containing
    the *y* data base line."""
    lenx = len(x)
    b_inds = np.arange(lenx)
    remi = [0]
    while len(remi) > 0 and len(b_inds) > lenx*(1. - rem):
        try:
            ma = smooth_ma(y[b_inds], wpn)
            d = abs(ma - y[b_inds])
            try:
                float(noise)
                remi = np.where((d > max(d)/remrate) & (d > noise))[0]
            except:
                remi = np.where((d > max(d)/remrate) & (d > noise[b_inds]))[0]
            if cbfun:
                cbfun(b_inds, ma, *cbfunargs)
            b_inds = np.delete(b_inds, remi, 0)
        except KeyboardInterrupt:
            break
    ma = smooth_ma(y[b_inds], wpn)
    spl = UnivariateSpline(x[b_inds], ma, k=k, s=0)
    if fulloutput:
        return spl(x), spl, b_inds
    else:
        return spl(x)



