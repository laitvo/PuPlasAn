#!/usr/bin/python
# -*- coding: utf-8 -*-

#abund.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020

"""
Abundances module
--------------------------

This module contains functions for estimation the abundances
of the species detected in the UV/vis spectrum of the LIBS plasma.
The computations are based on the CF-LIBS method.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as interpol
from scipy.integrate import odeint,quad
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma, airy
from scipy.optimize import curve_fit

kb = 1.38e-023
e = 1.602e-019
c_vac = 3e008
epsilon = 8.854e-012
h = 6.626e-034

    
def q(g0,gu,eu,T,T_stddev):
    """Calculates the partition function of the given species. Its value is evaluated as a partition sum corresponding to all possible transitions given for a species and its ionization state.
    
    Parameters
    ----------
    g0 : float
        ground state degeneracy of the species investigated
    gu : 1-D list/array
        array containing the upper level degeneracy of all possible transitions investigated
    eu : 1-D list/array
        array containing the upper level energy of all possible transitions investigated
        *eu* must be wavenumber in reciprocal centimeters
    T : float
        value of the excitation temperature of the given species within an ionization state given
    T_stddev : float
        mean square error of excitation temperature estimation
        
    Returns
    -------
    Q : float
        value of the partition function
    Q_stddev : float
        partition function uncertainty"""
    
    Q = g0+np.sum(gu*np.exp(-eu*(1.23984e-004*1.602e-019)/(kb*T)))
    
    
    q_max = g0+np.sum(gu*np.exp(-eu*(1.23984e-004*1.602e-019)/(kb*(T+T_stddev))))
    q_min = g0+np.sum(gu*np.exp(-eu*(1.23984e-004*1.602e-019)/(kb*(T-T_stddev))))
    
    Q_stddev = np.sqrt((((q_min-q)**2)+(q_max-q)**2)/(len(2)))

    return Q, Q_stddev
        
def c(mu,a,aij,gu,eu,T,Q,F=1.0):
    """Calculates the abundance of the given species based on pyrometric plot regression
    
    Parameters
    ----------
    mu : 1-D list/array
        array containing the peak positions
        *mu* must be wavelength in nanometers
	a : 1-D list/array
		array containing the integral intensity of the corresponding peaks
    aij : 1-D list/array
        array containing the Einstein's A coefficients of the investigated/detected transitions
		*aij* must be in reciprocal seconds
	gu: 1-D list/array
		array containing the upper level degeneracy of the investigated/detected transitions
    eu : 1-D list/array
        array containing the upper level energy of the investigated/detected transitions
		*eu* must be wavenumber in reciprocal centimeters
    T : float
        value of the excitation temperature of the given species within an ionization state given
    Q : float
        value of the partition function
    F : float
        opacity calibration parameter
        *F* must be a dimensionless number
        by default, *F* equals *1.0* for the needs of CF-LIBS analysis
        
    Returns
    -------
    c : float
        abundance of the given species within an ionization state given
    c_stddev : float
        mean square error of abundance estimation given for a defined species and ionization state"""
    
    nu = c_vac/(mu*1e-009)
    logI = np.log(a/(aij*gu*nu))

        
    c_T = lambda en,q: -1/(kb*T)*en+q
    
    c_,c_pc = curve_fit(c_T,eu*(1.23984e-004*1.602e-019),logI)
    c = 4*np.pi*q*np.exp(c_)/(h*F)
    c_stddev = (np.sqrt(np.abs(c_pc[1][1]))/c_)*c
    
    return c, c_stddev
