#!/usr/bin/python
# -*- coding: utf-8 -*-

#assign.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020

'''
Assignment module
'''


def assign(px1, px2, dx_max):
    '''This function assigns the closest values form *px1* and *px2*.

    Input(s):
       - px1: list or array (1d) of floats, peaks positions (experiment)
       - px2: list or array (1d) of floats, peaks positions (predictions)

    Output(s):
       - list of tuples with assigned indexes corresponding to the values
         in the input arrays *px1* and *px2* (e.g.: [(5,9), (6,12)...])'''
    inds = []
    for i1, px1i in enumerate(px1):
        select_aux = []
        for i2, px2i in enumerate(px2):
            d = abs(px1i - px2i) 
            if d < dx_max:
                select_aux.append((i1, i2, d))
        if len(select_aux) > 0:
            select_aux.sort(key=lambda x: x[2])
            inds.append((select_aux[0][0], select_aux[0][1]))
    return inds
