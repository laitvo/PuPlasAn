#!/usr/bin/python
# -*- coding: utf-8 -*-


#filters.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020



import copy
import warnings
import numpy as np


def smooth_ma(y, wpn):
    """Moving average smoothing.

    Parameters
    ----------
    y : 1-D list/array
        data to smooth
    wpn : int
        Number of the adjacent values averaged. *wpn* must be an odd number.
        If the integer is even, the *wpn* parameter is set to *wpn* + 1
        automatically and UserWarning is raised.

    Returns
    -------
    1-D numpy.array
        Array of the smoothed input data *y*."""
    if wpn < 1:
        wpn = 1
        warnings.warn("wpn must be an odd positive number" +
                      "(wpn set to %d)" % (wpn),
                      UserWarning)
    if wpn % 2 == 0:
        wpn += 1
        warnings.warn("wpn must be an odd positive number" +
                      " (wpn set to %d)" % (wpn),
                      UserWarning)
    yaux = np.concatenate((np.ones(int((wpn-1)/2))*y[0], np.asarray(copy.copy(y)),
                           np.ones(int((wpn-1)/2))*y[-1]))
    leny = len(y)
    ys = np.zeros(len(y))
    for i1 in range(wpn):
        ys += yaux[i1:i1+leny]
    return ys/wpn



