Reaction database ReadMe
========================

The reaction databases involved in deterministic reaction modelling should take the form

-------------------------------------------------------------------------------------------------------------------------
|reaction | rate coefficient formula | A | T0 | n | Ea | sigma | Eupp | Elow | gupp | glow | iE | Q(T) | Qf(T) | Z | a0 |
-------------------------------------------------------------------------------------------------------------------------

with a coma delimiter preferably set and decimal points used in floating point numbers.

1. *reaction* should be a standardised string formatted in a way similar to "A + 2 B = C+ + D-". Blank spaces must be typed around a reaction "+" sign to distinguish it from a charge label! "=" is used instead of a reaction arrow. Only elementary forward reactions are allowed as inputs.

2. *rate coefficient* should be a string containing either of the pre-set values, i. e. *Arrh* for Arrhenian reactions, *EC-exc*, *EC-deexc*, *EC-ion*, or *EC-rec*, or *EC-radrec* for particular electron reactions of a single heavy particle (Pietanza, L. D., Colonna, G., De Giacomo, A., & Capitelli, M. (2010). Kinetic processes for laser induced plasma diagnostic: A collisional-radiative model approach. Spectrochimica Acta - Part B Atomic Spectroscopy, 65(8), 616–626. https://doi.org/10.1016/j.sab.2010.03.012), *opt* for a reaction to be optimized or another explicit formula from the literature.

    2.1 *Arrh* input must be followed by the constants of modified Arrhenius equation as adopted from the NIST database (https://kinetics.nist.gov/kinetics/index.jsp). Ea should be an activation barrier of the particular process in eV. Spectroscopic characteristics may be marked by *-1* for empty values.

    2.2 Any *EC-* input must be followed by spectroscopic characteristics input. 
    *sigma* is a reaction cross section in cm2 and may be expresssed either as constants or as single-valued functions of electron energy. 
    *Eupp* and *Elow* are respectively the upper and lower level energies of the states involved in investigated reactions and given as formal wavenumbers in reciprocal centimeters. 
    *gupp* and *glow*, the degeneracies of such levels, should be dimensionless integers. *iE* is the ionization potential of the incident species given as formal wavenumber (cm-1). 
    *Q(T)* must be a string containing a partition function fitting equation, e. g. *3*T**3+2*T**2+T*1*. 
    *Qf(T)* must meet similar requirements and describe the partition function of a heavy particle produced if necessary. 
    *Z* is the net charge of the incident species given in integer multiples of e.
    *a0* is the Bohr radius of the incident species and must be given in meters.

    Modified Arrhenius equation parameters may be marked by *-1* for empty values.

    2.3 *opt* reactions to be optimized must be followed by an initial estimation of modified Arrhenius equation parameters (coming e. g. from the Eyring equation). Spectroscopic characterictics may be marked by *-1* for empty values. If an electron process is to be optimised, the *T0* row *-1* must be filled with *-1*. *A*, *n*, and *Ea* will then behave as free parameters of an expression *A*Te**n*exp(-Ea/(kb*Te))*.


3. If particular parameters are not known or not needed, *'-1'* must be filled in instead. 

4. Each database file should be named accordingly to the chemical system modelled (e. g. "CH4+N2") and given an ascii-friendly suffix. .CSV is preferred.
