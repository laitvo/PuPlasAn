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
Rate coefficients module
--------------------------

This module covers numerical calculations of electron reaction rate coefficients. Source functions are obtained from Pietanza, L. D., Colonna, G., De Giacomo, A., & Capitelli, M. (2010). Kinetic processes for laser induced plasma diagnostic: A collisional-radiative model approach. Spectrochimica Acta - Part B Atomic Spectroscopy, 65(8), 616–626. https://doi.org/10.1016/j.sab.2010.03.012
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

def vel_par(T_SI, T_SII, m=me, sample='SII'):
    """Depicts the Maxwell-Boltzmann distribution of the thermal electron velocity
    
    Parameters
    ----------
    T_SI : float
        excitation temperature of neutral species
        *T_SI* must be temperature in kelvins
    T_SII : float
        excitation temperature of once ionized species
        *T_SII* must be temperature in kelvins
    m : float
        mass of the reactive (transferred) species
        *m* must be mass in kilograms
        by default, *m* equals *me*
    sample : string
        Determines the species which is expected to be in a dynamic equilibrium with the reactive (transferred) species
        Default value is *'SII'*
       
    Returns
    -------   
    v : float
        most probable translation velocity given for thermal electrons in considered system
        *v* is estimated in m/s
    """ 

    if sample=='SII':
        T = T_SII

    if sample=='SI':
        T = T_SI

    v0 = np.sqrt(3.*kb*T/m)
    v = np.linspace(0.5*v0,1.5*v0,15000)
    N_v = 4.*np.pi*(v**2)*((m/(2.*np.pi*kb*T))**1.5)*np.exp(-m*(v**2)/(2.*kb*T))

    ind = np.argmin(abs(N_v-max(N_v)))

    return v[ind]

def distrib_electron(en,ne,T):
    """Approximates the electron energy distribution function based on experimental data
    
    Parameters
    ----------
    en: 1-D list/array
        array containing the energies of states within the species expected to be in a dynamic equilibrium with electrons
        *en* must be wavenumber in reciprocal centimetres
    ne : float
        electron density approximation
        *ne* must be electron density in reciprocal volume units
    T : float
        plasma temperature
        *T* must be temperature in kelvins
      
    Returns
    -------
    F : numpy.poly1d
        list of eedf fitting coefficients (polynomial)
    f_res : list
        list of residuals of the least-squares fit"""

    q_e = np.sum(np.exp(-en*(1.23984e-004*1.602e-019)/(kb*T)))
    N_e = []

    for i in range(len(en)):

        N = np.mean(n)*np.exp(-en[i]*(1.23984e-004*1.602e-019)/(kb*T))/q_e
        N = np.nan_to_num(N)
        N_e.append(N)
        i+=1

    f, f_res, f_rank, f_sv, f_rc = np.polyfit(en,N_e,6,full=True)
    F = np.poly1d(f)

    return F, f_res
        
def eedf(en,ne):
    """Depicts the Maxwell-Boltzmann electron energy distribution function
    
    Parameters
    ----------
    en: 1-D list/array
        array containing the energies of states within the species expected to be in a dynamic equilibrium with electrons
        *en* must be wavenumber in reciprocal centimetres
    ne : float
        electron density approximation
        *ne* must be electron density in reciprocal volume units
      
    Returns
    -------
    eedf : numpy.ndarray
        array containing the eedf
    F : numpy.poly1d
        list of eedf fitting coefficients (polynomial)
    f_res : list
        list of residuals of the least-squares fit"""    

    phi = np.mean(en)
    beta1 = (gamma(2.5)**1.5)*gamma(0.75)**-2.5
    beta2 = gamma(2.5)/gamma(1.5)
    eedf = (phi**-1.5)*beta1*np.exp(-en*beta2/phi)
    f, f_res, f_rank, f_sv, f_rc = np.polyfit(en,eedf,6,full=True)
    F = np.poly1d(f)

    return eedf, F, f_res
    
def rate_exc(eu, cross_exc, eedf, T):
    """Calculates the rate coefficient of electron excitation
    
    Parameters
    ----------
    eu: 1-D list/array
        array containing the upper level energies of the investigated states within the given species
        *en* must be wavenumber in reciprocal centimetres
    cross_exc : 1-D list/array
        array containing the electron excitation cross sections 
        *cross_exc* must be cross section in area units
    eedf : numpy.poly1d
        list of eedf fitting coefficients
    T : float
        plasma temperature
        *T* must be temperature in kelvins
      
    Returns
    -------
    K_exc : numpy.ndarray
        array containing the electron excitation rate coefficients
    K_exc_stddev : numpy.ndarray
        array containing the electron excitation rate coefficients estimation error"""  

    v = vel_par(T,T)
    K_exc = []
    K_exc_stddev = []

    for i in range(len(eu)):

        alpha = cross_exc[i]
        f = lambda E,alpha,v: alpha*eedf(E)*v
        K = scipy.integrate.quad(f,eu[i],+np.inf,args=(alpha,v,))[0]
        K_error = scipy.integrate.quad(f,eu[i],+np.inf,args=(alpha,v,))[1]
        K_exc.append(K)
        K_exc_stddev.append(K_error)

    return K_exc, K_exc_stddev
            
            
def rate_deexc(gu, eu, gl, el, T, cross_exc, eedf):
    """Calculates the rate coefficient of electron impact de-excitation
    
    Parameters
    ----------
    gu : 1-D list/array
        array containing the upper level degeneracies of the investigated states within the given species
    eu : 1-D list/array
        array containing the upper level energies of the investigated states within the given species
        *eu* must be wavenumber in reciprocal centimetres
    gl : 1-D list/array
        array containing the lower level degeneracies of the investigated states within the given species
    el : 1-D list/array
        array containing the lower level energies of the investigated states within the given species
        *el* must be wavenumber in reciprocal centimetres
    T : float
        plasma temperature
        *T* must be temperature in kelvins
    cross_exc : 1-D list/array
        array containing the electron excitation cross sections 
        *cross_exc* must be cross section in area units
    eedf : numpy.poly1d
        list of eedf fitting coefficients
      
    Returns
    -------
    K_deexc : numpy.ndarray
        array containing the electron impact de-excitation rate coefficients
    K_deexc_stddev : numpy.ndarray
        array containing the electron impact de-excitation rate coefficients estimation error"""  

    v = vel_par(T,T)

    K_deexc = []
    K_deexc_stddev = []

    for j in range(len(eu)):

        alpha = gu[j]*eu[j]/(gl[j]*el[j])*cross_exc[j]
        f = lambda E,alpha,v: alpha*eedf(E)*v
        K = scipy.integrate.quad(f,eu[j],+np.inf,args=(alpha,v,))[0]
        K_error = scipy.integrate.quad(f,eu[j],+np.inf,args=(alpha,v,))[1]
        K_deexc.append(K)
        K_deexc_stddev.append(K_error)

    return K_deexc, K_deexc_stddev
        
def rate_ion(gu, eu, T, cross_ion, eedf):
    """Calculates the rate coefficient of electron impact ionization
    
    Parameters
    ----------
    gu : 1-D list/array
        array containing the upper level degeneracies of the investigated states within the given species
    eu : 1-D list/array
        array containing the upper level energies of the investigated states within the given species
        *eu* must be wavenumber in reciprocal centimetres
    T : float
        plasma temperature
        *T* must be temperature in kelvins
    cross_ion : 1-D list/array
        array containing the electron impact ionization cross sections 
        *cross_ion* must be cross section in area units
    eedf : numpy.poly1d
        list of eedf fitting coefficients
      
    Returns
    -------
    K_ion : numpy.ndarray
        array containing the electron impact ionization rate coefficients
    K_ion_stddev : numpy.ndarray
        array containing the electron impact ionization rate coefficients estimation error""" 

    v = vel_par(T,T)
    K_ion = []
    K_ion_stddev = []

    for k in range(len(eu)):

        alpha = cross_ion[k]
        f = lambda E,alpha,v: alpha*eedf(E)*v
        K = scipy.integrate.quad(f,eu[k],+np.inf,args=(alpha,v,))[0]
        K_error = scipy.integrate.quad(f,eu[k],+np.inf,args=(alpha,v,))[1]
        K_ion.append(K)
        K_ion_stddev.append(K_error)

    return K_ion, K_ion_stddevs
        
def rate_rec(eu, gu, gl, T, cross_ion, en_ion, q_f, eedf):
    """Calculates the rate coefficient of three-body recombination
    
    Parameters
    ----------
    eu : 1-D list/array
        array containing the upper level energies of the investigated states within the given species
        *eu* must be wavenumber in reciprocal centimetres
    gu : 1-D list/array
        array containing the upper level degeneracies of the investigated states within the given species
    gl : 1-D list/array
        array containing the lower level degeneracies of the investigated states within the given species
    T : float
        plasma temperature
        *T* must be temperature in kelvins
    cross_ion : 1-D list/array
        array containing the electron impact ionization cross sections 
        *cross_ion* must be cross section in reciprocal area units
    en_ion : float
        ionization energy of the given state
        *en_ion* must be wavenumber in reciprocal centimetres
    q_f : float
        partition function of the final state of the species (e. g. neutral atom for a single-ionized atomic ion)
    eedf : numpy.poly1d
        list of eedf fitting coefficients
      
    Returns
    -------
    K_rec : numpy.ndarray
        array containing the three-body recombination rate coefficients
    K_rec_stddev : numpy.ndarray
        array containing the three-body recombination rate coefficients estimation error""" 

    q = np.sum(np.array(gu)*np.exp(-np.array(eu)*(1.23984e-004*1.602e-019)/(kb*T)))
    K_eq = (q**2.)/q_f
    v = vel_par(T,T)
    K_rec = []
    K_rec_stddev = []

    for l in range(len(eu)):

        alpha = cross_rec[l]*((eu[l]/(np.abs(eu[l]-en_ion)))*(gu[l]/gl[l])*np.exp(-en_ion*(1.23984e-004*1.602e-019)/(kb*T))/K_eq)
        f = lambda E,N,alpha,v: alpha*eedf(E)*v
        K = scipy.integrate.quad(f,eu[l],+np.inf,args=(alpha,v,))[0]
        K_error = scipy.integrate.quad(f,eu[l],+np.inf,args=(alpha,v,))[1]
        K_rec.append(K)
        K_rec_stddev.append(K_error)

    return K_rec, K_rec_stddev
        
def rate_radrec(eu, gu, mu, T, Z, a0, en_ion, eedf):
    """Calculates the rate coefficient of radiative recombination
    
    Parameters
    ----------
    eu : 1-D list/array
        array containing the upper level energies of the investigated states within the given species
        *eu* must be wavenumber in reciprocal centimetres
    gu : 1-D list/array
        array containing the upper level degeneracies of the investigated states within the given species
    mu : 1-D list/array
        array containing the transition peak positions within the given species
        *mu* must be wavelength in nanometers
    T : float
        plasma temperature
        *T* must be temperature in kelvins
    Z : float/integer
        ionization state of the species (1, 2, ... n)
    a0 : float
        covalent radius (Bohr radius in case of atoms) of the species
        *a0* must be dimension in metres  
    cross_ion : 1-D list/array
        array containing the electron impact ionization cross sections 
        *cross_ion* must be cross section in area units
    en_ion : float
        ionization energy of the given state
        *en_ion* must be wavenumber in reciprocal centimetres
    eedf : numpy.poly1d
        list of eedf fitting coefficients
      
    Returns
    -------
    K_radrec : numpy.ndarray
        array containing the radiative recombination rate coefficients
    K_tarec_stddev : numpy.ndarray
        array containing the radiative recombination rate coefficients estimation error""" 

    v = vel_par(T,T)
    K_radrec = []
    K_radrec_stddev = []

    for m in range(len(eu)):

        alpha = (np.pi*(a0**2.)*32.*(Z**4.)*Ry**2.)/(3.*np.sqrt(3.)*(137.**3.)*h*c/(mu[m]*1e-009))*(en_ion/(np.abs(en_ion-eu[m])))**-1.5
        f = lambda E,alpha,v: alpha*eedf(E)*v
        K = scipy.integrate.quad(f,eu[m],+np.inf,args=(alpha,v,))[0]
        K_error = scipy.integrate.quad(f,eu[m],+np.inf,args=(alpha,v,))[1]
        K_radrec.append(K)
        K_radrec_stddev.append(K_error)

    return K_radrec, K_radrec_stddev
