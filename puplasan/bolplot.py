#!/usr/bin/python
# -*- coding: utf-8 -*-


#bolplot.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020


"""
Boltzmann plot module
--------------------------

This module contains functions which determine the excitation and
free electron temperature of the system using the experimental UV/VIS spectrum.
All the computations are based on the approximation of the local thermodynamic equilibrium (LTE).
"""

import warnings
import numpy as np
import pylab as pl
import scipy
from scipy.optimize import curve_fit

h = 6.626e-034
c_vac = 3e008
kb = 1.38e-023

	
def logbp(mu,a,aij,gu):
    """Calculates the Boltzmann's logarithm.

    Parameters
    ----------
    mu : 1-D list/array
        array containing the peak positions.
		*mu* must be wavelength in nanometers
    a : 1-D list/array
        array containing the integral intensity of the corresponding peaks
    aij : 1-D list/array
        array containing the Einstein's A coefficients of the investigated/detected transitions
		*aij* must be in reciprocal seconds
	gu: 1-D list/array
		array containing the upper level degeneracy of the investigated/detected transitions

    Returns
    -------
    
    logI : numpy.array
        Boltzmann's logarithm of the integral intensity used for the estimation
        excitation temperature estimation"""
        
    nu = c_vac/(mu*1e-009)
    logI = np.log(a/(aij*gu*nu))
    
    return logI
	

def T(eu,logI):
    """Performs the pyrometric plot regression to estimate excitation temperature

    Parameters
    ----------
    eu : 1-D list/array
        array containing the upper level energy of the investigated/detected transitions
		*eu* must be wavenumber in reciprocal centimeters
    logI : 1-D list/array
        array containing the values of the Boltzmann's logarithm

    Returns
    -------
    
    T : float
        excitation temperature for the given species and its transitions defined
        *T* is defined in kelvins
    """
    
    K, cov = np.polyfit(((eu)*(1.23984e-004*1.602e-019)),logI,1,cov=True)
    
    T = -1/(K[0]*kb)
    T_stddev = (np.sqrt(np.abs(cov[0][0]))/K[0])*T
    
    return T, T_stddev
	
def T_electron(eu,a,T_SI,T_SII,T_SI_stddev,T_SII_stddev):
    """Performs an empiric calculation of the electron temperature expected to fulfill the equilibrium between neutral and single-ionized species
    
    Parameters
    ----------
    eu : 1-D list/array
        array containing the upper level energy of the investigated/detected transitions
		*eu* must be wavenumber in reciprocal centimeters
	a : 1-D list/array
		array containing the integral intensity of the corresponding peaks
	T_SI : float
		value of the excitation temperature of neutral state of the investigated species
        *T_SI* and *T_SII* must be given in consistent units        
	T_SII : float
		value of the excitation temperature of single-ionized state of the investigated species
        *T_SI* and *T_SII* must be given in consistent units  
	T_SI_stddev : float
		mean square error of the excitation temperature of neutral state of the investigated species
        *T_SI_stddev* and *T_SII_stddev* must be given in consistent units         
	T_SII_stddev : float
		mean square error of the excitation temperature of single-ionized state of the investigated species
        *T_SI_stddev* and *T_SII_stddev* must be given in consistent units    
        
    Returns
    -------
    T_e : float
        expected electron temperature value"""
    
    e = eu[np.where(a==max(a))[0][0]]*1.23984e-004*1.602e-019
    a = 0.47045563170594129
    b = -0.69735896613283599
    
    T_e = np.abs((-e/kb)/(np.log(np.abs((a*(-e/(kb*T_SI))+b*(-e/(kb*T_SII)))/(a+b)))))
    T_e_stddev = np.abs((-e/1.38e-023)/(np.log(np.abs((a*(-e/(1.38e-023*(T_SI+T_SI_stddev)))+b*(-e/(1.38e-023*(T_SII+T_SII_stddev))))/(a+b))))-(-e/1.38e-023)/(np.log(np.abs((a*(-e/(1.38e-023*T_SI))+b*(-e/(1.38e-023*T_SII)))/(a+b)))))

    
    return T_e, T_e_stddev
	
def temp(theta_min, theta_max, spec, lines_all, C, Q, px, py):
    """Performs simpler nonlinear numerical optimization of an experimental spectrum on generalized LTE conditions using which a single excitation temperature is obtained

    Parameters
    ----------
    theta_min : float
        infimum of a temperature range within which the optimum si sought for
		*theta_min* must be temperature in kelvins
    theta_max : float
        supremum of a temperature range within which the optimum si sought for
		*theta_max* must be temperature in kelvins
	spec : list of strings
		list containing labels of the investigated species in a form of "S I" etc.  
    lines_all : list of arrays
        list of arrays containing the database inputs for all investigated/detected species
	C : 1-D list/array
		list of abundances corresponding to the investigated species of an ionization state given
	Q : 1-D list/array
		list of partition functions corresponding to the investigated species of an ionization state given 
    px : 1-D list/array
        array containing the positions of the detected peaks
        *px* must be wavelength in nanometers
    py : 1-D list/array
        array containing integral intensities of the detected peaks    

    Returns
    -------
    T : float
        excitation temperature
        note that if *T* equals *theta_min* or *theta_max* the test was probably not decisive
    T_stddev : float
        mean square error of the excitation temperature"""

    def assig_database(species,lines):
        i = 0
        mu = []
        aij = []
        eu = []
        gu = []

        for i in range(len(lines)):

            index = lines[i][1]
            m = species[index,0]
            a = species[index,1]
            en = species[index,2]
            g = species[index,3]
            mu.append(m)
            aij.append(a)
            eu.append(en)
            gu.append(g)

        return mu,aij,eu,gu

    theta = np.linspace(theta_min, theta_max, 10000)
    stddev = []
    linelist_all = {spec[i]:lines_all[i] for i in range(len(spec))}

    for m in range(len(theta)):

        error = 0

        for o in range(len((spec))):
        
            sp = linelist_all[spec[o]]
            wl = sp[:,0]
            lines = ass.assign(px,wl,d_max)
            
            if lines==[]:
            
                error=0

            else: 
            
                mu,aij, eu, gu = np.array(assig_database(sp,lines))
                a = assig_peak(py,lines)
                sig = C[o]*aij*gu*h*c_vac/(4*np.pi*Q[o]*mu*1e-009)*np.exp(-eu*1.23984e-004*1.602e-019/(kb*theta[m]))
                err = np.nan_to_num(np.sum((sig/max(sig)-a/max(a))**2)/len(sig))
                error+=err
                

        stddev.append(error)

    T = np.mean(theta[np.where(np.diff(np.diff(stddev))==min(np.diff(np.diff(stddev))))[0]])
    
    try:
    
        T_stddev = np.mean(stddev[np.where(np.diff(np.diff(stddev))==min(np.diff(np.diff(stddev))))[0]])*T/max(a)
        
    except:
    
        T_stddev = T/max(a)
        
    return T, T_stddev

def T_discrim(eu, logI, T0, deltaT, deltaT_min, deltaT_max):   
    """Performs a simpler distance point discrimination analysis in the Boltzmann plane defined for a given species and ionization state to obtain a statistically enhanced excitation temperature value

    Parameters
    ----------
    eu : 1-D list/array
        array containing the upper level energy of the investigated/detected transitions
		*eu* must be wavenumber in reciprocal centimeters
    logI : 1-D list/array
        array containing the values of the Boltzmann's logarithm
	T0 : float
		initial assumption on the excitation temperature value
        *T0* must be temperature in kelvins 
    deltaT : float
        temperature difference limit
        *deltaT* must be temperature in kelvins
    deltaT_min : float
        minimal temperature increment by which to enhance the Boltzmann plot
        *deltaT_min* must be temperature in kelvins
    deltaT_min : float
        minimal temperature increment by which to enhance the Boltzmann plot
        *deltaT_min* must be temperature in kelvins     

    Returns
    -------
    T : float
        excitation temperature
        note that if *T* equals *theta_min* or *theta_max* the test was probably not decisive
    T_stddev : float
        mean square error of the excitation temperature"""
    
    p0 = -1/(kb*T0)
    f = lambda x, q: p0*x + q
    popt, pcov = curve_fit(f,eu*(1.23984e-004*1.602e-019),logbp)
    
    T = []
    T_stddev = []
    i = 0
    
    slither = np.linspace(deltaT_max,deltaT_min,deltaT_max)
    
    while i<len(slither) and len(eu)>1:
        
        p0_neutral_slupp = -1/(kb*(T0_neutral+slither[i]))
        p0_neutral_sllow = -1/(kb*(T0_neutral-slither[i]))
        logbp_low = p0_neutral_sllow*np.asarray(eu)*(1.23984e-004*1.602e-019)+popt
        logbp_upp = p0_neutral_slupp*np.asarray(eu)*(1.23984e-004*1.602e-019)+popt
        boltz = [(en_uppi,logi) for ii, (en_uppi, logi) in enumerate(zip(eu,logbp)) if logbp[ii]<logbp_upp[ii] and logbp[ii]>logbp_low[ii]]
    
        if boltz !=[]:
    
            eu, logbp = zip(*boltz)
        
        else:
    
            eu = eu
            logbp = logbp
        
        t,t_stddev = bp.T(np.asarray(eu),np.asarray(logbp))
        
        if np.abs(t-T0_neutral)<delta:
        
            T.append(t)
            T_stddev.append(t_stddev)
        i+=1

    if T!=[]:
    
        T = T[np.where(T_stddev==min(T_stddev))][0]
        T_stddev = T_stddev[np.where(T_stddev==min(T_stddev))][0]
        
    else:
    
        T = t
        T_stddev = t_stddev
        
        
    return T, T_stddev
