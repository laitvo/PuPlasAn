# PuPlasAn (**Pu**lsed **Plas**ma **An**alyser)
## Public ReadMe

The program is responsible for a complete numerical analysis of pulsed plasma spectra, i. e. filtering (`filters.py`), estimating noise level (`noise.py`) and spectral baseline (`bline.py`), and detecting spectral lines and/or bands (`pdetect.py`). The peaks detected may then undergo deterministic or stochastic numerical fitting of line or band profiles (`pfit.py`). Optical depth and spectral line broadening issues may be dealt with by using `broad.py`.

Next, several procedures based on applying the Boltzmann law can be used to evaluate plasma distribution temperature(s) with `bolplot.py`. Spectroscopic diagnostics of the transitions observed may be defined ad hoc or using a class `pbase.py`. 
Based on similar relations and approaches, `abund.py` performs stepwise elemental and speciation analyses of an LTE pulsed plasma. Alternatively, such properties may be extracted from a solution to radiative transfer equations (`rte.py`).  

Finally, `rate.py` and `mechanism.py` calculate reaction rate coefficients based on user-defined data (`rdbase.csv` with a ReadMe file) and cover simpler kinetic modelling of the plasma systems investigated with time-resolved spectroscopic methods.

The program was developed by Petr Kubelík, Ph. D., and Vojtěch Laitl in 2022 at the J. Heyrovský Institute of Physical Chemistry, Czech Academy of Sciences. From there on, it is distributed under Apache License 2.0. All details to particular functions and program modules are to be found in the above code scripts and in the example section.
