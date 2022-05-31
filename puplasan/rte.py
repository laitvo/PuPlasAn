import numpy as np
#import pylab as pl
import scipy
from scipy.optimize import curve_fit
import scipy.integrate
from scipy.optimize import fsolve
import copy
import warnings
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma
from scipy.integrate import odeint
from . import broad as sbm
from . import pfit as pf

"""
RTE module
--------------------------

This module contains functions which cover the solution of the Radiative Transfer Equation. The approximations valid for of an isotropic/homogeneous medium are exploited.
"""

h = 6.626e-034
c = 3e008
kb = 1.38e-023
sigma = 5.6704e-008
Ry = 10973731.6
Na = 6.022e023
a0 = 0.529177
eps0 = 8.854e-012
me = 9.109e-031

profiles = {'gauss': pf.gauss, 'lorentz': pf.lorentz, 'pseudovoigt': pf.pseudovoigt}
convert = {'cm-1': lambda nu: 1e002*nu/c, 'nm': lambda nu: 1e009*c/nu}

def planck(nu, T):
    """Plots the black-body spectral radiance profile parametrized by constant temperature
    
    Parameters
    ----------
    nu : float OR numpy.array
        emission frequency
        *nu* must be frequency in reciprocal seconds
    T : float
        equilibrium temperature
        *T* must be temperature in kelvins
       
    Returns
    -------   
    Bv : numpy.array
        black-body spectral radiance profile
    """ 

    Bv = (2.*h*nu**3/(c**2))*1./(np.exp(h*nu/(kb*T))-1.)

    return Bv

def boltz(N0, gj, Ej, T, Q_T):
    """Returns a Boltzmann equilibrium abundance of absorbing/emitting species
    
    Parameters
    ----------
    N0 : float
        total number density of the species
        *N0* must be number density in reciprocal cubic meters
    gj : numpy.array
        array containing lower level degeneracies of the investigated species
    Ej : numpy.array
        array containing lower level energies of the investigated species
        *Ej* must be formal wavenumber in reciprocal centimeters
    T : float
        equilibrium temperature
        *T* must be temperature in kelvins
    Q_T : numpy.poly1d
        list of Q(T) polynomial fitting coefficients
       
    Returns
    -------   
    N : numpy.array
        array evaluating the number densities against given energy levels
    """ 
    Q = Q_T(T)
    N = N0*gj*np.exp(-Ej*(1.23984e-004*1.602e-019)/(kb*T))/Q

    return N

def Aij_rv(nu, Ju, Jl, D):
    """Calculates an Einstein A coefficient for given rovibrational transitions based on their dipole moments
    
    Parameters
    ----------
    nu : numpy.array
        frequency of investigated transition(s)
        *nu* must be frequency in reciprocal seconds
    Ju : numpy.array
        upper level rotational quantum number
    Jl : numpy.array
        lower level rotational quantum number
    D : float
        dipole moment of given transitions
        *D* must be dipole moment in coulomb-meters

    Returns
    -------   
    Aul : numpy.array
        array containing resultant Einstein coefficients
        *Aul* is Einstein coefficient in reciprocal seconds    
    """

    Aul = []

    for i in range(len(nu)):

        S = max([Ju[i], Jl[i]])
        Ai = ((16.*(np.pi**3)*(nu[i])**3)/(3.*eps0*h*(c**3)))*(S/(2.*Ju[i]+1.))*D**2
        Aul.append(Ai)

    Aul = np.array(Aul)

    return Aul

def abs_Aij(nu, Aij):
    """Returns an absorption cross section calculated from the Einstein A coefficient
    
    Parameters
    ----------
    nu : float OR numpy.array
        frequency of investigated transition(s)
        *nu* must be frequency in reciprocal seconds
    Aij : float OR numpy.array
        Einstein A coefficient of investigated transition(s)
        *Aij* must be frequency in reciprocal seconds

    Returns
    -------   
    sigma : float OR numpy.array
        absorption cross section of the given species
        *sigma* is cross section in square meters
    """

    Fij = (2.*h*nu**3)/(c**2)
    Bji = Aij/Fij
    sigma = (h*nu/c)*Bji

    return sigma

def abs_muij(D2):
    """Returns an absorption cross section calculated from the transition dipole moment
    
    Parameters
    ----------
    D2 : float OR numpy.array
        squared transition dipole moment of invenstigated transtions
        *D2* must be squared dipole moment in atomic units

    Returns
    -------   
    sigma : float OR numpy.array
        absorption cross section of the given species
        *sigma* is cross section in square meters
    """

    D2 = np.array(D2)*(3.33564e-030)**2.

    sigma = 2.*(np.pi/(h*c))*D

    return sigma

def attenuation(sigma, N):
    """Returns an attenuation coefficient of either species
    
    Parameters
    ----------
    sigma : float OR numpy.array
        absorption cross section of investigated transition(s)
        *sigma* must be a cross section in square meters
    N : float OR numpy.array
        number density ascribed to the lower energy level of investigated transition(s)
        *N* must be number density in reciprocal cubic meters

    Returns
    -------   
    alpha : float OR numpy.array
        attenuation coefficient
        *alpha* is given in reciprocal meters
    """

    alpha = sigma*N

    return alpha

def emission(alpha, nu, T):
    """Returns an emission coefficient of either species
    
    Parameters
    ----------
    alpha : float OR numpy.array
        attenuation coefficient
        *alpha* must be given in reciprocal meters
    nu : float OR numpy.array
        frequency of investigated transition(s)
        *nu* must be frequency in reciprocal seconds
    T : float
        equilibrium temperature
        *T* must be temperature in kelvins

    Returns
    -------   
    j : float OR numpy.array
        emission coefficient
        *j* is given in J per square meter
    """

    Bv = planck(nu, T)

    j = alpha*Bv
     
    return j

def doppler_fwhm(nu, m, T, units='cm-1'):
    """Calculates on the formal Doppler broadening of given absorption/emission transitions
    
    Parameters
    ----------
    nu : float OR numpy.array
        frequency of given transition(s)
        *nu* must be frequency in reciprocal seconds
    m : float
        nominal mass of the investigated species
        *m* must be mass in kilograms
    T : float
        equilibrium temperature
        *T* must be temperature in kelvins
    units : string
        string defining the units of desider output
        *'cm-1'* and *'nm'* are allowed inputs
        by default, *units* equals *'cm-1'*

    Returns
    -------   
    mu : float OR numpy.array
        peak centre of the investigated transition(s)
        *mu* is given in specified *units*
    fwhm : float OR numpy.array
        formal FWHM of the investigated transition(s) absorption/emission coefficients
        *fwhm* is given in specified *units*
    """
    
    if units == 'cm-1':
    
        mu = nu/c
        fwhm = mu*(np.sqrt(8.*kb*T*np.log(2)/(m*c**2)))
    
    else:
    
        mu = c/nu
        fwhm = (mu**2)*np.sqrt(8.*kb*T*np.log(2)/(m*c**4))
    
        mu, fwhm = mu*1e009, fwhm*1e009

    return mu, fwhm

def rte_block(wn, I0, T, ell, nu, alpha, fwhm, broadening, units='cm-1'):
    """Performs a simpler RTE solution in a homogeneous isotropic medium of temperature *T* and optical length *ell*
    
    Parameters
    ----------
    wn : numpy.array
        frequency axis of a spectrum measured at zero position
    I0 : numpy.array
        nominal intensity of a source (x = 0)
        *I0* must be intensity in watts per square meter
    T : float
        equilibrium temperature
        *T* must be temperature in kelvins
    ell : float
        optical length of the medium
        *ell* must be length in meters
    nu : numpy.array
        frequency of given transition(s)
        *nu* must be frequency in reciprocal seconds   
    alpha : numpy.array
        attenuation coefficient of given transitions
        *alpha* must be given in reciprocal meters  
    broadening : string
        string defining a chosen broadening profile
        *'gauss'*, *'lorentz'* and *'pseudovoigt'* are allowed values        
    units : string
        string defining the units of desider output
        *'cm-1'* and *'nm'* are allowed values
        by default, *units* equals *'cm-1'*

    Returns
    -------   
    mu : float OR numpy.array
        peak centre of the investigated transition(s)
        *mu* is given in specified *units*
    fwhm : float OR numpy.array
        formal FWHM of the investigated transition(s) absorption/emission coefficients
        *fwhm* is given in specified *units*
    """
    
    j = emission(alpha, nu, T)
    
    alpha_nu = np.zeros(len(wn))
    j_nu = np.zeros(len(wn))
    
    for i in range(len(alpha)):
    
        ai_nu = profiles[broadening](wn, convert[units](nu[i]), alpha[i], fwhm[i])
        ji_nu = profiles[broadening](wn, convert[units](nu[i]), j[i], fwhm[i])
        
        alpha_nu += ai_nu
        j_nu += ji_nu

    I_ell = I0*np.exp(-alpha_nu*ell) + j_nu*np.exp(-alpha_nu*ell)*ell

    return I_ell
