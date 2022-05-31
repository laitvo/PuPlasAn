#!/usr/bin/python
# -*- coding: utf-8 -*-

#pdetect.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020

"""
Peak detection module
---------------------

This module contains functions (especially the *detect* function)
for detection peaks in the experimental data (e.g. UV/VIS spectrum).
"""

import numpy as np
import copy

def detectTop_inds(y, ptype='up'):
    """Selects the indexes of *y* where the top of a peak is detected.

    Parameters
    ----------
    y : 1-D list/array
        array of input data.
    ptype : str
        only acceptable values are "up" (detection of local maxima)
        and "down" (detection of local minima)

    Returns
    -------
    numpy.array
        array containing the indexes of *y* where the top
        of a peak is detected"""
    dy = np.diff(y)
    dy1 = dy[:-1]
    dy2 = dy[1:]
    dymul = dy1 * dy2
    dysub = dy1 - dy2
    dymulinds = np.where(dymul < 0.0)[0]
    if ptype == 'up':
        dysubinds = np.where(dysub > 0.0)[0]
    elif ptype == 'down':
        dysubinds = np.where(dysub < 0.0)[0]
    else:
        raise Exception('invalid ptype (must be set to "up" or "down",' +
                        ' actual value == %s)' % str(ptype))
    intersec = np.intersect1d(dymulinds, dysubinds) + 1
    return intersec


def detectThr_inds(y, thr, ptype='up'):
    """Selects the indexes of *y* where the values are higher (ptype="up")
    or lower (ptype="down") than *thr*.

    Parameters
    ----------
    y : 1-D list/array
        array of input data.
    thr : float/array
        selection threshold (if array - it is x-dependent threshold)
    ptype : str
        only acceptable values are "up" (selects indexes of values
        higher than *thr*)
        and "down" (selects indexes of values lower than *thr*)

    Returns
    -------
    numpy.array
        array containing the indexes of *y* where the values
        are higher (ptype="up") or lower (ptype="down") than *thr*"""
    if ptype == 'up':
        inds = np.where(y > thr)[0]
    elif ptype == 'down':
        inds = np.where(y < thr)[0]
    else:
        raise Exception('invalid ptype (must be set to "up" or "down",' +
                        ' actual value == %s)' % str(ptype))
    return inds


def detect(x, y, noise, snr_thr, ptype='up', fulloutput=False):
    """Detect peaks' positions and highs in the input data arrays.

    Parameters
    ----------
    x : 1-D list/array
        array of independent input data. Must be increasing.
    y : 1-D list/array
        array of dependent input data, of the same length as *x*.
    noise : float/array
        noise of the *y* data (if array - it is x-dependent noise)
    snr_thr : float
        the high threshold of the peak selection
        is calculated as *noise* \* *snr_thr*
    ptype : str
        only acceptable values are "up" (detection of local maxima)
        and "down" (detection of local minima)
    fulloutput : bool
        Influence the output (see the *Returns* section).
        Default value is *False*.

    Returns
    -------
    If *fulloutput* is set to *True* numpy.array of local extrema positions
    (values from *x*), numpy.array of local extrema highs (values from *y*)
    and numpy.array of *x*'s indexes where the local extrema are dedected
    is returned.

    Else numpy.array of local extrema positions (values from *x*) and
    numpy.array of local extrema highs (values from *y*) is returned."""
    thr = noise * snr_thr
    pinds = np.intersect1d(detectThr_inds(y, thr, ptype=ptype),
                           detectTop_inds(y, ptype=ptype))
    if fulloutput:
        return x[pinds], y[pinds], pinds
    else:
        return x[pinds], y[pinds]





