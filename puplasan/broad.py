#!/usr/bin/python
# -*- coding: utf-8 -*-


#broad.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020

"""
Broadening module
--------------------------

This module contains functions which describe the broadening of the
spectral lines detected in the UV/vis spectrum and compute the basic parameters
related to the line broadening, i.e. the Stark width, the electron density and the self-absorption coefficient.
"""

import numpy as np
import pylab as pl
import scipy
from scipy.optimize import curve_fit
from scipy.special import comb
from numpy.polynomial.hermite import hermval
from scipy.integrate import quad, cumtrapz
from scipy.optimize import curve_fit
import math

try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial

h = 6.626e-034
c = 3e008
kb = 1.38e-023
me = 9.109e-031
Ry = 10973731.6
eps = 8.854e-012
e = 1.602e-019

def sw(a0,a1,a2,T):
    """
    Calculates the theoretical Stark widths based on the database input and a parametrical recursive relation

    log(w) = a0 + a1 log(T) + a2 log{2}(T)

    with T being a termodynamic temperature for which the database defines w

    Parameters
    ----------
    a0 : float OR 1-D list/array
        absolute member of the recursive relation
        *a0* should be read directly from the program database
    a1 : float OR 1-D list/array
        linear logaritm memember of the recursive relation
        *a1* should be read directly from the program database
    a2 : quadratic logarithm memember of the recursive relation
        *a2* should be read directly from the program database

    Returns
    ----------
    width_stark : float OR 1-D list/array
        assigned tabulated Stark widths in nanometers

    """

    width_stark = np.exp(a0+a1*np.log(T)+a2*np.log(T)**2)

    return width_stark
    
    
def assign_stark(fwhm,r,x):
    """Ascribes the FWHM parameters to the tabulated Stark widths
    
    Parameters
    ----------
    fwhm: 1-D list/array
        array containing the FWHM of the peaks related to the lorentzian profile
        *fwhm* must be in nanometers
    r : 1-D list/array
        array containing the peak positions of the lines with tabulated Stark widths
        *r* must be wavelength in nanometers
    x : 1-D list/array
        array containing the peak positions
        *x* must be wavelength in nanometers
      
    Returns
    -------
    sigma_stark : numpy.array
        the array containing the ascribed Stark widths"""
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    sigma_stark = []
    i = 0
    while i<len(r):        
        sigma0 = sigma[np.where(np.abs(x-r[i])==min(np.abs(x-r[i])))[0][0]]
        sigma_stark.append(sigma0)
        i+=1
    return sigma_stark
    
def stark2ne(fwhm_stark,width_stark,fulloutput=False):
    """Calculates the electron density
    
    Parameters
    ----------
    fwhm_stark: 1-D list/array
        array containing the FWHM of detected peaks 
        related to the lorentzian profile
        *FWHM* must be in nanometers
    width_stark : 1-D list/array
        array containing the assigned tabulated Stark widths
    full_output : bool
        Influences the output (see the *Returns* section)
        Default value is *False*
        
    Returns
    -------
    if full_output=False (default)
        Ne : float
            value of the electron density.
            *Ne* is estimated in reciprocal cubic centimeters
        error_Ne : float
            uncertainty of the electron density (*error_Ne*)
            *error_Ne* is estimated in reciprocal cubic centimeters
            
    if full_output=True
        Ne : float
            value of the electron density
            *Ne* is estimated in reciprocal cubic centimeters
        error_Ne : float
            uncertainty of the electron density (*error_Ne*)
            *error_Ne* is estimated in reciprocal cubic centimeters
   
        If *fulloutput* is *True*, draws additionally a plot of the Stark widths over FWHM whose
        linear regression is related to the electron density
    """
    fwhm_stark = np.insert(fwhm_stark,0,0)
    width_stark = np.insert(width_stark,0,0)
    fit, cov = np.polyfit(fwhm_stark,width_stark,1,cov=True)
    k = fit[0]
    Ne = np.abs(10**16/(2*k))
    uncert = np.sqrt(np.abs(cov[0][0]))/k
    error_Ne = Ne*uncert
    if fulloutput:
        pl.plot(fwhm_stark,width_stark,'bo'), pl.plot(fwhm_stark,fit_fn(fwhm_stark),'b')
        pl.xlabel('FWHM (nm)')
        pl.ylabel('Stark width (nm)')
        pl.show()
    return Ne, error_Ne
     
def ne2ws(fwhm,Ne,error_Ne,full_output=False):
    """Estimates the unknown Stark widths
    
    Parameters
    ----------
    fwhm : 1-D list/array
        array containing the FWHM of detected peaks.
        *FWHM* must be in nanometers.
    Ne : float
        value of the electron density.
        *Ne* must be in reciprocal cubic centimeters.
    error_Ne : float
        value of the electron density estimation error.
        *error_Ne* must be in reciprocal cubic centimeters.
    full_output : bool
        Influences the output (see the *Returns* section)
        Default value is *False*       
        
    Returns
    -------
    if full_output=False (default)
        ws : numpy.array
            array containing the estimated Stark widths
            *ws* is estimated in nanometers
        error_ws : numpy.array
            uncertainty of the Stark widths estimation
            *error_ws* is estimated in nanometers
            
    if full_output=True
        ws : numpy.array
            array containing the estimated Stark widths
            *ws* is estimated in nanometers
        error_ws : numpy.array
            uncertainty of the Stark widths estimation    
            *error_ws* is estimated in nanometers
            
        If *fulloutput* is *True*, draws additionally a plot of the Stark widths over FWHM
        including the experimental data.
    """
    ws = fwhm/2*1/(Ne*10**-16)
    ws_min = fwhm/2*1/((Ne-error_Ne)*10**-16)
    ws_max = fwhm/2*1/((Ne+error_Ne)*10**-16)
    ws_i = [ws_min,ws_max]
    error_ws = np.sqrt((((ws_min-ws)**2)+(ws_max-ws)**2)/(len(ws)))
    if full_output:
        fwhm = np.insert(fwhm,0,0)
        ws = np.insert(ws,0,0)
        fit = np.polyfit(fwhm,ws,1)
        fit_fn = np.poly1d(fit)
        pl.plot(fwhm,ws,'bo'),pl.plot(fwhm,fit_fn(fwhm),'b')
        pl.xlabel('FWHM (nm)')
        pl.ylabel('Stark width (nm)')
        pl.show()
    return ws, error_ws
    
def self_absorption(fwhm,ws,Ne,error_Ne,error_ws):
    """Calculates the self-absorption coefficient of the given peaks
    
    Parameters
    ----------
    fwhm : 1-D list/array
        array containing the FWHM of the peaks
        *FWHM* must be in nanometers
    ws : 1-D list/array
        array containing the Stark widths
        *ws* must be in nanometers
    Ne : float
        value of the electron density
        *Ne* must be in reciprocal cubic centimeters
    error_Ne : float
        value of the electron density estimation error.
        *error_Ne* must be in reciprocal cubic centimeters
    error_ws : numpy.array
        uncertainty of the Stark widths estimation    
        *error_ws* must be in nanometers
       
    Returns
    -------
    sa : numpy.array
        self-absorption coefficients of the given peaks
    error_sa : numpy.array
        uncertainty of the Stark coefficients estimation
    """ 
    ws_min = ws-error_ws
    ws_max = ws+error_ws
    sa = (fwhm*10**16/2*ws/Ne)**1/2
    sa_min = (fwhm*10**16/2*ws_min/(Ne-error_Ne))**1/2
    sa_max = (fwhm*10**16/2*ws_max/(Ne+error_Ne))**1/2
    error_sa = np.sqrt((((sa_min-sa)**2)+(sa_max-sa)**2)/(len(sa)))
    return sa, error_sa

