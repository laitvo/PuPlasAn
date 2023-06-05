#!/usr/bin/python
# -*- coding: utf-8 -*-


#pfit.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2022


import numpy as np

"""
Peak profile fitting module
---------------------------

Contains functions for fitting the peaks profiles. The peak fitting
is performed by *rsopt* function which is based on random rearch algorithm
and is able to fit a model composed by multiple (possibly overlapping)
peaks represented by an instance of the :class:`puplasan.pfit.PeakModel` class.
This function is applicable also for single peak fitting.
"""

def rsopt(obfun, bounds, msteps=50, fsteps=100, tsteps=500, focus=0.5,
          ftol=[3, 1.e-5], obfun_args=(), obfun_kwargs={}, callback=None,
          callback_args=(), callback_kwargs={}):
    if fsteps >= tsteps:
        raise Exception('tsteps must be greater than fsteps ' +
                        '(current values: fsteps = %d, ' +
                        'tsteps = %d)' % (fsteps, tsteps))
    iter_bounds = bounds[:]
    mcounter = 0
    fcounter = 0
    tcounter = 0
    ftolcounter = 0
    fold = float('inf')
    while tcounter <= tsteps:
        mcounter += 1
        fcounter += 1
        tcounter += 1
        pars = []
        for (ibmin, ibmax), (bmin, bmax) in zip(iter_bounds, bounds):
            maxx = min(bmax, ibmax)
            minx = max(bmin, ibmin)
            #(b - a) * np.random.random_sample() + a; b>a 
            pars.append((maxx - minx) * np.random.random_sample() + minx)
        fnew = obfun(pars, *obfun_args, **obfun_kwargs)
        if fnew < fold:
            if abs(fnew - fold) < ftol[1]:
                ftolcounter += 1
            fold = fnew
            best_pars = pars[:]
            fcounter = 0
            tcounter = 0
            mcounter = 0
            if callback:
                callback(best_pars, *callback_args)
            if ftolcounter > ftol[0]:
                break #convergence
        if mcounter >= msteps:#move
            iter_bounds_new = []
            for (ibmin, ibmax), (bmin, bmax), bp in zip(iter_bounds, bounds, best_pars):
                r = ((ibmax - ibmin))/2.
                maxb = min(bmax, bp + r)
                minb = max(bmin, bp - r)
                iter_bounds_new.append((minb, maxb))
            iter_bounds = iter_bounds_new[:]
            mcounter = 0
        if fcounter >= fsteps:#focus
            iter_bounds_new = []
            for (ibmin, ibmax), (bmin, bmax), bp in zip(iter_bounds, bounds, best_pars):
                r = ((ibmax - ibmin)*focus)/2.
                maxb = min(bmax, bp + r)
                minb = max(bmin, bp - r)
                iter_bounds_new.append((minb, maxb))
            iter_bounds = iter_bounds_new[:]
            fcounter = 0
    return best_pars, fold


def area_estim(high, fwhm):
    """Estimates the peak are from the peak high and fwhm.

    Parameters
    ----------
    high : float or numpy.array
        peak high
    fwhm : float or numpy.array
        peak fwhm

    Returns
    -------
    area
        peak area estimated as *high* \* *fwhm*"""
    return high * fwhm


def detect_peak_groups(px, d):
    """Generates groups of peaks which are so close each to other that they
    can be overlapped.

    Parameters
    ----------
    px : 1-D list/array
        array of the peak positions (obtained by a peak picking method)
    d : float/callable
        if float, it is the distance threshold (peaks which are closer to each
        other are grouped), if callable, it is a function with one input
        parameter (the peak position (float)) which returns the distance
        threshold expressed as a function of the independent variable
        corresponding to *px*

    Returns
    -------
    groups : list of lists of tuples
        list describing the grouping of the peaks accordind their mutual
        distances, the format is following:
        [[(i0, v0), (i1, v1)...], [(in, vn), (in+1, vn+1)...]...], where each
        nested list represent a group of mutually close peaks and each tuple
        in these lists contains index of an element of the input parameter
        *px* and the corresponding value."""
    px.sort()
    groups = [[(0, px[0])]]
    if hasattr(d, '__call__'):
        for i1, pxi in enumerate(px[1:], 1):
            #appends next peak to the last group
            if (pxi - groups[-1][-1][1]) < d(pxi):
                #creates a new group and insert the next peak
                groups[-1].append((i1, pxi))
            else:
                groups.append([(i1, pxi)])
    else:
        for i1, pxi in enumerate(px[1:], 1):
            #appends next peak to the last group
            if (pxi - groups[-1][-1][1]) < d:
                #creates a new group and insert the next peak
                groups[-1].append((i1, pxi))
            else:
                groups.append([(i1, pxi)])
    return groups


def gauss(x, mu, a, fwhm):
    """Calculates the Gaussian profile function from the input parameters.

    Parameters
    ----------
    x : 1-D list/array
        array of independent variable values. Must be increasing.
    mu : float
        peak center position
    a : float
        peak area
    fwhm : float
        peak full width at half maximum

    Returns
    -------
    numpy.array
        array containing the Gaussian peak profile
        corresponding to the indepentent variable *x*"""
    sigma = fwhm/2.35482
    return ((1./(sigma * np.sqrt(2.*np.pi))) *
            np.exp(-0.5 * ((x - mu)/sigma)**2)) * a


def lorentz(x, mu, a, fwhm):
    """Calculates the Lorentzian profile function from the input parameters.

    Parameters
    ----------
    x : 1-D list/array
        array of independent variable values. Must be increasing.
    mu : float
        peak center position
    a : float
        peak area
    fwhm : float
        peak full width at half maximum

    Returns
    -------
    numpy.array
        array containing the Lorentzian peak profile
        corresponding to the indepentent variable *x*"""
    gamma = fwhm/2.
    return ((1./np.pi) * ((0.5*gamma)/((x - mu)**2 + (0.5*gamma)**2))) * a


def pseudovoigt(x, mu, a, fwhm, s=0.2):
    """Calculates the Pseudo-Voigt (weighted sum of Gaussian and Lorentzian
       profiles) profile function from the input parameters.

    Parameters
    ----------
    x : 1-D list/array
        array of independent variable values. Must be increasing.
    mu : float
        peak center position
    a : float
        peak area
    fwhm : float
        peak full width at half maximum
    s : float <0,1>
        peak shape parameter (if *s*==0 the profile is pure Gaussian,
        if *s*==1 the profile is pure Lorentzian)

    Returns
    -------
    numpy.array
        array containing the Lorentzian peak profile
        corresponding to the indepentent variable *x*"""
    g = gauss(x, mu, 1., fwhm)
    l = lorentz(x, mu, 1., fwhm)
    return ((1. - s)*g + s*l) * a


class PeakModel:
    """Provides methods to model single/multi peak profiles."""
    profiles = {'gauss': gauss,
                'lorentz': lorentz,
                'pseudovoigt': pseudovoigt}

    def __init__(self, mu, area, fwhm, pars=(), profile='gauss'):
        """
        Parameters
        ----------
        mu : list/tuple of floats
            positions of the centers of the peaks included in the profile model
        area : list/tuple of floats
            areas of the peaks included in the profile model
        fwhm : list/tuple of floats
            full widths in half maxima of the peaks included
            in the profile model
        pars : list of lists/tuples of floats
            other single peak profile parameters (each nested tuple
            corresponds to a specific parameter of all peaks of the model)
            used by the selected line profile function (e.g. the shape
            parameter used by the *pseudovoigt* function)
        profile : str
            a string corresponding to one of single peak profile functions
            ("gauss", "lorentz", "pseudovoigt"), this profile is used for
            all peaks of the model"""
        self.profile = profile
        self.mu = mu
        self.area = area
        self.fwhm = fwhm
        self.pars = pars
        if any([len(mu) != len(area), len(mu) != len(fwhm)]):
            raise Exception('the length of mu, area and fwhm must be equal')
        self.pnum = len(mu)

    def get_flat_pars(self):
        """Returns internal peak model parameters in a single flat list
        (used as an input partameter of the *calc_fit_error* method)

        Returns
        -------
        pars : list of floats
            model parameters in a single flat list with the following format:
            [par_i_peak_j, par_i+1_peak_j, par_i+2_peak_j...
            par_i_peak_j+1, par_i+1_peak_j+1... par_n_peak_m]
            where "par_i_peak_j" refers to the ith parameter
            of the jth peak of the profile model"""
        if self.pars:
            return ([el for tup in zip(self.mu, self.area, self.fwhm,
                                      *self.pars) for el in tup])
        else:
            return ([el for tup in zip(self.mu, self.area,
                                      self.fwhm) for el in tup])

    def set_flat_pars(self, pars):
        """Sets the internal peak parameters of the profile model

        Parameters
        ----------
        pars : list of floats
            model parameters in a single flat list with the following format:
            [par_i_peak_j, par_i+1_peak_j, par_i+2_peak_j...
            par_i_peak_j+1, par_i+1_peak_j+1... par_n_peak_m]
            where "par_i_peak_j" refers to the ith parameter
            of the jth peak of the profile model, the peak number
            as well as the numper of the parameters of each peak must be
            in agreement with the internal peak model"""
        parn = int(len(pars)/self.pnum)
        aux = list(zip(*[pars[i:i+parn] for i in range(0, len(pars), parn)]))
        self.mu = aux[0]
        self.area = aux[1]
        self.fwhm = aux[2]
        if len(aux) > 3:
            self.pars = aux[3:]
        else:
            self.pars = ()

    def gen_profile(self, x, full_output=False):
        """Returns the profile of multiple (possibly overlapped) peaks.
        The resulting profile is calculated as a sum of several
        peak-like profiles.

        Parameters
        ----------
        x : 1-D list/array
            array of independent input data. Must be increasing.
        full_output : bool
            if True: returns array containing the resulting multi-peak profile
            as well as the profiles of the individual functions
            (e.g. gaussians)
            else: returns array containing the resulting multi-peak profile

        Returns
        -------
        numpy.array
            array containing the resulting multi-peak profile
            corresponding to the indepentent variable *x*
            (and a list of the profiles of the individual functions
            (e.g. gaussians) (only if the full_output is set to True))"""
        y = np.zeros(len(x))
        parts = []
        if self.pars:
            pars = zip(self.mu, self.area, self.fwhm, *self.pars)
        else:
            pars = zip(self.mu, self.area, self.fwhm)
        for parsi in pars:
            prof = PeakModel.profiles[self.profile](x, *parsi)
            y += prof
            if full_output:
                parts.append(prof)
        if full_output:
            return y, parts
        return y

    def calc_fit_error(self, pars_flat, datax=[], datay=[]):
        """Calculates the objective function value used by
        a fitting function

        Parameters
        ----------
        pars_flat : list/tuple of floats
            profile model parameters in the following format:
            [par_i_peak_j, par_i+1_peak_j, par_i+2_peak_j...
            par_i_peak_j+1, par_i+1_peak_j+1... par_n_peak_m]
            where "par_i_peak_j" refers to the ith parameter
            of the jth peak of the profile model, the peak number
            as well as the numper of the parameters of each peak must be
            in agreement with the internal peak model (this list can be
            generated by *get_flat_pars* method) 
        datax : 1-D list/array
            array of independent variable values. Must be increasing.
        datay : 1-D list/array
            array of dependent variable values.

        Returns
        -------
        err : float
            the difference between the model profile and the *datay* parameter,
            the value is calculated es follows:
            np.sqrt(sum((self.gen_profile(datax) - datay)**2))

        .. warning::

           by calling this method the internal model parameters are rewriten
           according to the input parameter *pars_flat*"""
        self.set_flat_pars(pars_flat)
        return np.sqrt(sum((self.gen_profile(datax) - datay)**2)/
                       float(len(datax)))


def peak_model(px, py, spe, mode, pl_range= [200., 212., -0.07, 0.6], step=12):
    """Builds a fit of experimental spectrum based on stochastic modeling of small areas optima

    Parameters
    ----------
    px : 1-D list/array
        array of the peak positions (obtained by a peak picking method)
    py : 1-D list/array
        array containing integral intensities of the detected peaks
    spe : numpy.array
        array containing the experimental spectrum as sorted onto two lists
        *[wl, signal]*
        *"wl"* must be given in units consistent with *px*
    mode: string
		mode in which to operate the fit procedure
        values *'isolated'* (to treat all peaks as isolated transitions) and *'ifc'* (to explicitly include instrumental broadening function into spectral patterns) are allowed
    pl_range : list
        list defining the initial spectral area to search on and additional constants as defined above
        by default, *pl_range* equals *[200., 212., -0.07, 0.6]*
    step : float
        iteration step by which the length of a fitted spectrum is enhanced
        *step* must be unit-consistent with *px*
        by default, *step* equals 12


    Returns
    -------
    
    fit_spect : numpy.array
        y-axis of the fitted/simulated spectrum"""

    if mode == 'isolated':
        
        fwhm_estim = 0.005
        profile_ = 'pseudovoigt'
        
    if mode == 'ifc':
    
        fwhm_estim = 0.05
        profile_ = 'lorentz'
        

    fit_spect = []

    while pl_range[1]<max(px) and pl_range[1] != []:
    
        ind1 = np.where(spe[:,0] > pl_range[0])[0][0]
        ind2 = np.where(spe[:,0] > pl_range[1])[0][0]
        indg1 = np.where(px > pl_range[0])[0][0]
        indg2 = np.where(px > pl_range[1])[0][0]
        groups = detect_peak_groups(px[indg1:indg2], 0.05)
        fit_spect0 = np.zeros(len(spe[ind1:ind2,0]))
            
        for gi in groups:
            
            ind1_fit = np.where(spe[:,0] > gi[0][1] - 0.1)[0][0]
            ind2_fit = np.where(spe[:,0] > gi[-1][1] + 0.1)[0][0]
            bounds = []
            mu = []
            area = []
            fwhm = []
                
            for pki in gi:
                
                a_estim = area_estim(py[pki[0]], fwhm_estim)
                mu.append(pki[1])
                area.append(a_estim)
                fwhm.append(fwhm_estim)
                bounds.extend([(pki[1] - 0.02, pki[1] + 0.02),
                    (a_estim/10., a_estim*10.),
                    (fwhm_estim/5., fwhm_estim*3.)])
            peak = PeakModel(mu, area, fwhm, pars=(), profile=profile_)
            best_pars, fold = rsopt(peak.calc_fit_error, bounds, msteps=20, fsteps=500,
                            tsteps=5000, focus=0.6, ftol=[3, 1.e-6],
                            obfun_args=(spe[ind1_fit:ind2_fit,0],
                                        spe[ind1_fit:ind2_fit,1]),
                            obfun_kwargs={}, callback=None, callback_args=(),
                            callback_kwargs={})
            fit_spect0 += peak.gen_profile(spe[ind1:ind2,0])

        fit_spect = np.insert(fit_spect,len(fit_spect),fit_spect0)
        pl_range[0] += step
        pl_range[1] += step

    if len(fit_spect) < len(spe[:,0]):
        fit_spect = np.insert(fit_spect,len(fit_spect),np.zeros(len(spe[:,0])-len(fit_spect)))

    else:
        if len(fit_spect) > len(spe[:,0]):
            fit_spect = fit_spect[:len(spe[:,0])]

        else:
            pass

    return fit_spect
