import numpy as np
import pylab as pl
import scipy
from scipy.optimize import curve_fit
import scipy.integrate
from scipy.integrate import odeint
from scipy.optimize import fsolve
import copy
import warnings
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma
from . import broad as sbm
from sympy import *


"""
Reaction mechanism module
--------------------------

This module contains functions responsible for assembling and integrating a reaction mechanism.
"""

h = 6.626e-034
c = 3e008
kb = 1.38e-023
sigma = 5.6704e-008
Ry = 10973731.6
Na = 6.022e023
R = kb*Na
p0 = 1e005
me = 9.109e-031


def stoichiometric_matrix(rdbase_path, species_list):
    """Prints a stoichiometric matrix of a reaction mechanism described

    Parameters
    ----------
    rdbase_path : string
        file path to a filled-in database
        the database mut be an ascii-text file and must meet the README-defined format
    species_list : string
        list of all species involved in the mechanism
      
    Returns
    -------
    S : numpy.ndarray
        stoichiometric matrix of the mechanism
    S_lhs : numpy.ndarray
        (pseudo)stoichiometric matrix describing the left-hand sides of the chemical transformations involved
    """

    reac = np.genfromtxt(rdbase_path, skip_header=1, usecols=0, delimiter=',', dtype='str')

    S = [np.zeros(len(reac))]*len(species_list)
    S_lhs = [np.zeros(len(reac))]*len(species_list)    

    for i in range(len(reac)):

        reac_lhs = reac.split(' = ')[0]
        reac_rhs = reac.split(' = ')[1]

        reactants = reac_lhs.split(' + ')
        products = reac_rhs.split(' + ')

        for j in range(len(reactants)):

            rj = np.where(species_list == reactants[j])[0][0]
            
            try:

                S[rj][i] = -float(reactants[i].split()[0])
                S_lhs[rj][i] = -float(reactants[i].split()[0])

            except:

                S[rj][i] = -1.
                S_lhs[rj][i] = -1.           
            
        for k in range(len(products)):

            pk = np.where(species_list == products[k])[0][0]
            
            try:

                S[pk][i] = float(reactants[i].split()[0])

            except:

                S[pk][i] = 1.   

    return S, S_lhs


def rate_coefficients(rdbase_path):
    """Converts reaction database data to a function calculating reaction rate coefficients

    Parameters
    ----------
    rdbase_path : string
        file path to a filled-in database
        the database mut be an ascii-text file and must meet the README-defined format
      
    Returns
    -------
    lambda_k : python lambda function
        function calculating the rate coefficients of all reactions involved
        *T*, *Te*, and *ne* are required as numerical inputs
        *T* given in kelvins is the (translational) plasma temperature
        *Te* given in kelvins is the free-electron temperature
        *ne* is the electron number density and must be given in units corresponding to *k*
    """

    rc = np.genfromtxt(rdbase_path, skip_header=1, usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], delimiter=',', dtype='str')
    rt = np.array(rc[:,0], 'str')
    A = np.array(rc[:,1], 'float')
    T0 = np.array(rc[:,2], 'float')
    n = np.array(rc[:,3], 'float')
    Ea = np.array(rc[:,4], 'float')
    sigma = np.array(rc[:,5], 'str')
    Eu = np.array(rc[:,6], 'float')
    El = np.array(rc[:,7], 'float')
    gu = np.array(rc[:,8], 'float')
    gl = np.array(rc[:,9], 'float')
    iE = np.array(rc[:,10], 'float')
    Q_T = np.array(rc[:,11], 'str')
    Qf_T = np.array(rc[:,12], 'str')
    Z = np.array(rc[:,13], 'float')
    a0 = np.array(rc[:,14], 'float')    

    T = Symbol('T', positive=True)
    Te = Symbol('Te', positive=True)
    ne = Symbol('ne', positive=True)
    E = Symbol('E')

    phi = kb*Te
    beta1 = (gamma(2.5)**1.5)*gamma(0.75)**-2.5
    beta2 = gamma(2.5)/gamma(1.5)
    eedf = (ne**-1.5)*beta1*exp(-E*(1.23984e-004*1.602e-019)*beta2/phi)    
    v = 100.*sqrt(3.*kb*Te/me)
    Nv = 4.*np.pi*(v**2)*((me/(2.*np.pi*kb*Te))**1.5)*exp(-m*(v**2)/(2.*kb*Te))

    k_ = []
    init_guess = []
    opt = 0

    for i in range(len(rt)):

        if rt[i] == 'Arrh':

            ki = A[i]*((T/T0)**n)*exp(-Ea[i]/(kb*T))

        if rt[i].split('-')[0] == 'EC':

            if rt[i].split('-')[1] == 'exc' or rt[i].split('-')[1] == 'ion':
                alpha = sigma[i]

            if rt[i].split('-')[1] == 'deexc':
                alpha = sigma[i]*gu[i]*Eu[i]/(gl[j]*El[j])
 
            if rt[i].split('-')[1] == 'rec':
                Keq = (sympify(Q_T[i])**2)/sympify(Qf_T[i])
                alpha = sigma[i]*(Eu[i]/abs(Eu[i]-iE[i]))*(gu[i]/gl[i])*exp(-iE[i]*(1.23984e-004*1.602e-019)/(kb*T))/Keq

            if rt[i].split('-')[1] == 'radrec':
                nu = c*(Eu[i]-El[i])
                alpha = (pi*(a0[i]**2)*32.*(Z[i]**4)*(Ry**2)/(3.*sqrt(3.)*(137.**3)*h*nu))*(iE[i]/abs(Eu[i]-iE[i]))**-1.5

            f = sympify(alpha)*eedf*v
            ki = integrate(f, (E, Eu[i], oo))

        if rt[i] == 'opt':
            init_guess.append([A[i],n[i],Ea[i]])

            if T0[i] == 0.:

                ri = ropt[3*opt]*(Te)**(ropt[3*opt+1])*np.exp(-ropt[3*opt+2]/(kb*Te))

            else:

                ri = ropt[3*opt]*(T/T0[i])**(ropt[3*opt+1])*np.exp(-ropt[3*opt+2]/(kb*T))

            opt += 1

        else:

            ki = sympify(rt[i])

        k_.append(ki)

    lambda_k = lambdify((T,Te,ne), k_)

    if init_guess != []:
        init_guess = np.concatenate(init_guess)

    else:
        init_guess = None    

    return lambda_k, init_guess


def reac(c0, t, S, S_lhs, k):
    """Assembles a reaction mechanism in a form of an integrable ODE system

    Parameters
    ----------
    c0 : 1-D list/array
        array containing the initial concentrations of the species involved
    t : numpy.array
        array containing the time axis along which the system is integrated
    S : numpy.ndarray
        stoichiometric matrix of the mechanism
    S_lhs : numpy.ndarray
        (pseudo)stoichiometric matrix describing the left-hand sides of the chemical transformations involved
    k : 1-D list/array
        array containing the rate coefficients of the chemical transformations involved
        *k*, *t*, and *N* must appear in consistent units
      
    Returns
    -------
    d : numpy.array
        reaction rate vector to be integrated
    """ 

    r = k
    
    for i in range(len(S_lhs[0])):

        ri = S_lhs[:,i]
    
        for j in range(len(ri)):

            if ri[j] != 0:
                r[i] *= c0[j]**ri[j]

            else:
                pass

    d = np.dot(S,r)

    return d


def reac_min(init_guess, lambda_k, T, Te, Ne, c0, t, t_i, S, S_lhs, species, k, species_ref, chi_ref):
    """Takes the filtered database input and assembles a function for rate coefficient calculation based on experimental data
    
    Parameters:
    ----------
    init_guess : 1-D list/array
        list of initial guess values of rate coefficient parameters for the reactions to be optimized
    lambda_k : python lambda function
        function calculating the rate coefficients of all reactions involved
        *T*, *Te*, and *ne* are required as numerical inputs
        *T* given in kelvins is the (translational) plasma temperature
        *Te* given in kelvins is the free-electron temperature
        *ne* is the electron number density and must be given in units corresponding to *k*
    T : float
        kinetic temperature of the system
        *T* must be temperature in kelvins
    Te : float
        electron temperature of the system
        *Te* must be temperature in kelvins
    Ne : float
        electron number density of the system
        *Ne* must be unit-consistent with the rate coefficients dimension (preferably reciprocal cubic centimeters)
    c0 : 1-D list/array
        array containing the initial concentrations of the species involved
        *c0* must be unit-consistent with the rate coefficients dimension (preferably reciprocal cubic centimeters)
    t : 1-D list/array
        time axis along which to integrate the model
        *t* must be time in seconds
    t_i : 1-D list/array
        time stamps corresponding to measurements/estimations of *chi_ref*
        *t_i* must be time in seconds
    S : numpy.ndarray
        stoichiometric matrix of the mechanism
    S_lhs : numpy.ndarray
        (pseudo)stoichiometric matrix describing the left-hand sides of the chemical transformations involved
    species : list
        ordered list of species involved in the kinetic model
    k : list
        list of rate coefficient formulae or references for the former
    species_ref : string
        label of a reference species the mixing ratio of which is optimized (preferably buffer species or those in large excesss)
    chi_ref : 1-D list/array
        list of reference mixing ratios of the species investigated within optimized reactions

    Returns:
    ----------
    norm_rsd : float
        norm of residues to be optimized
    """

    opt = []

    for i in range(len(k)):

        if k[i] == 'opt':
            opt = np.append(opt,i)

        else:
            pass

    ii = []

    for j in range(len(t_i)):

        ii = np.append(ii,np.argmin(abs(t-t_i)))

    ref = np.where(species==species_ref)[0][0]

    rc = lamba_k(T, Te, Ne, init_guess)
    N = odeint(reac, c0, t, args=(S, S_lhs, rc,))

    rsd = N[:,opt][ii]/N[:,ref][ii] - chi_ref
    rsd[np.isnan(rsd)] = 0.
    rsd[np.isinf(rsd)] = 0.

    norm_rsd = np.linalg.norm(rsd,np.inf)

    return norm_rsd


def reac_opt(init_guess, lambda_k, T, Te, Ne, c0, t, t_i, S, S_lhs, species, species_ref, chi_ref, campaign, rdbase_path):
    """Optimizes a reaction mechanism and prints a refined database file
    
    Parameters:
    ----------
    init_guess : 1-D list/array
        list of initial guess values of rate coefficient parameters for the reactions to be optimized
    lambda_k : python lambda function
        function calculating the rate coefficients of all reactions involved
        *T*, *Te*, and *ne* are required as numerical inputs
        *T* given in kelvins is the (translational) plasma temperature
        *Te* given in kelvins is the free-electron temperature
        *ne* is the electron number density and must be given in units corresponding to *k*
    T : float
        kinetic temperature of the system
        *T* must be temperature in kelvins
    Te : float
        electron temperature of the system
        *Te* must be temperature in kelvins
    Ne : float
        electron number density of the system
        *Ne* must be unit-consistent with the rate coefficients dimension (preferably reciprocal cubic centimeters)
    c0 : 1-D list/array
        array containing the initial concentrations of the species involved
        *c0* must be unit-consistent with the rate coefficients dimension (preferably reciprocal cubic centimeters)
    t : 1-D list/array
        time axis along which to integrate the model
        *t* must be time in seconds
    t_i : 1-D list/array
        time stamps corresponding to measurements/estimations of *chi_ref*
        *t_i* must be time in seconds
    S : numpy.ndarray
        stoichiometric matrix of the mechanism
    S_lhs : numpy.ndarray
        (pseudo)stoichiometric matrix describing the left-hand sides of the chemical transformations involved
    species : list
        ordered list of species involved in the kinetic model
    species_ref : string
        label of a reference species the mixing ratio of which is optimized (preferably buffer species or those in large excesss)
    chi_ref : 1-D list/array
        list of reference mixing ratios of the species investigated within optimized reactions
    campaign : string
        label of the experimental campaign solved
    rdbase_path : string
        file path to a filled-in database
        the database mut be an ascii-text file and must meet the README-defined format

    Returns:
    ----------
    k_optimized : list
        list of optimized rate coefficient formulae or references for the former
    A_optimized : list
        list of optimized Arrhenius pre-exponential factors (floating point numbers)
    n_optimized : list
        list of optimized Arrhenius power parameters (floating point numbers)
    Ea_optimized : list
        list of optimized activation energies of individual reactions (floating point numbers)
        *Ea* is activation energy in electronvolts
    """

    rc = np.genfromtxt(rdbase_path, skip_header=1, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], delimiter=',', dtype='str')
    reac = np.array(rc[:,0],'str')
    rt = np.array(rc[:,1], 'str')
    A = np.array(rc[:,2], 'float')
    T0 = np.array(rc[:,3], 'float')
    n = np.array(rc[:,4], 'float')
    Ea = np.array(rc[:,5], 'float')
    sigma = np.array(rc[:,6], 'str')
    Eu = np.array(rc[:,7], 'float')
    El = np.array(rc[:,8], 'float')
    gu = np.array(rc[:,9], 'float')
    gl = np.array(rc[:,10], 'float')
    iE = np.array(rc[:,11], 'float')
    Q_T = np.array(rc[:,12], 'str')
    Qf_T = np.array(rc[:,13], 'str')
    Z = np.array(rc[:,14], 'float')
    a0 = np.array(rc[:,15], 'float')    

    opt = []

    for i in range(len(k)):

        if k[i] == 'opt':
            opt.append(i)

        else:
            pass

    bds_r = []

    for i in range(0,len(init_guess),3):

        bds_r.append((0.25*init_guess[i],2.*init_guess[i]))
        bds_r.append((-2.*init_guess[i+1],2.*init_guess[i+1]))
        bds_r.append((0.25*init_guess[i+2],2.*init_guess[i+2]))
        
    bds_r = tuple(bds_r)
    par_opt = minimize(reac_min,init_guess,args=(lambda_k, T, Te, c0, t, t_i, S, S_lhs, species, k, species_ref, chi_ref,),bounds = bds_r).x

    for i in range(len(opt)):

        if T0[i] == 0.:

            rt[i] = '%.2e*Te**(%.2e)*exp(-%.2e/(kb*Te))'%(par_opt[3*i],par_opt[3*i+1],par_opt[3*i+2])
            A[i] = -1
            #T0[i] = -1
            n[i] = -1
            Ea[i] = -1

        else:

            rt[i] = 'Arrh'
            A[i] = par_opt[3*i]
            #T0[i] = 300.
            n[i] = par_opt[3*i+1]
            Ea[i] = par_opt[3*i+2]*e

    head = ['reaction','rate coefficient','A','T0','n','Ea','sigma','Eupp','Elow','gupp','glow','iE','Q(T)','Qf(T)','Z','a0']
    dbase = zip(reac, rt, A, T0, n, Ea, sigma, Eupp, Elow, gupp, glow, iE, Q_T, Qf_T, Z, a0)
    
    with open(campaign+'.csv', 'w') as csvfile:

        writer = csv.writew(csvfile)
        [writer.writerow(head)]
        [writer.writerow(r) for r in dbase]

    k_optimized = k
    A_optimized = A
    #T0_optimized = T0
    n_optimized = n
    Ea_optimized = Ea

    return k_optimized, A_optimized, n_optimized, Ea_optimized
