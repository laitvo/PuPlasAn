#!/usr/bin/python
# -*- coding: utf-8 -*-

#pbase.py
#Petr Kubelík, Ph. D. et all at J. Heyrovsky Institute of Physical Chemistry, CAS, Czech Republic, 2020


"""
Module for management of the spectral lines database.
"""

import sqlite3
import numpy as np


class PBase:
    """This class is resposible for managing the spectral lines and atomic
    energy transitions data (e.g. level energies, g-factors,
    transition Eistein coefficients, line positions, intensities etc.)

    The data are stored in a sqlite3 database file. 

    Tables' structure:

    .. table:: specie

       +-------------+-----------+--------------------------------------------+
       | column name | data type | descrip./exmple                            |
       +=============+===========+============================================+
       | id          | integer   | specie id number                           |
       +-------------+-----------+--------------------------------------------+
       | name        | text      | specie name (e.g. "Fe I")                  |
       +-------------+-----------+--------------------------------------------+
       | charge      | integer   | specie charge (e.g. 0, 1, 2...)            |
       +-------------+-----------+--------------------------------------------+

    .. note::

        following SQL constraints are applied:
       - id is a primary key, autoincrement
       - UNIQUE(name, charge) ON CONFLICT IGNORE

    .. table:: level

       +-------------+-----------+--------------------------------------------+
       | column name | data type | descrip./exmple                            |
       +=============+===========+============================================+
       | id          | integer   | level id number                            |
       +-------------+-----------+--------------------------------------------+
       | id_specie   | integer   | id of the specie                           |
       +-------------+-----------+--------------------------------------------+
       | el_conf     | text      | electron configuration (e.g. 3s2.3p)       |
       +-------------+-----------+--------------------------------------------+
       | term        | text      | spectroscopic term (e.g. 2P*)              |
       +-------------+-----------+--------------------------------------------+
       | j           | text      | total ang. momentum quant. num. (e.g. 1/2) |
       +-------------+-----------+--------------------------------------------+
       | en          | real      | level energy [cm-1]                        |
       +-------------+-----------+--------------------------------------------+
       | g           | real      | g-factor                                   |
       +-------------+-----------+--------------------------------------------+
       | ref         | text      | reference                                  |
       +-------------+-----------+--------------------------------------------+

    .. note::

        following SQL constraints are applied:
        - UNIQUE(id_specie, el_conf, term, j) ON CONFLICT REPLACE
        - id is a primary key, autoincrement
        - id_specie is a foreign key

    .. table:: transition

       +--------------+-----------+-------------------------------------------+
       | column name  | data type | descrip./exmple                           |
       +==============+===========+===========================================+
       | id           | integer   | transition id number                      |
       +--------------+-----------+-------------------------------------------+
       | id_specie    | integer   | id of the specie                          |
       +--------------+-----------+-------------------------------------------+
       | id_level_low | integer   | id of the lower energy level              |
       +--------------+-----------+-------------------------------------------+
       | id_level_upp | integer   | id of the upper energy level              |
       +--------------+-----------+-------------------------------------------+
       | wavelength   | real      | transtion wavelength [nm]                 |
       +--------------+-----------+-------------------------------------------+
       | aij          | real      | Einstein coefficient [s-1]                |
       +--------------+-----------+-------------------------------------------+
       | ref          | text      | references                                |
       +--------------+-----------+-------------------------------------------+

    .. note::

        following SQL constraints are applied:
        - UNIQUE(id_specie, id_level_low, id_level_upp) ON CONFLICT REPLACE
        - id is a primary key, autoincrement
        - id_specie, id_level_low and id_level_upp are foreign keys

    .. table:: peak

       +--------------+-----------+-------------------------------------------+
       | column name  | data type | descrip./exmple                           |
       +==============+===========+===========================================+
       | id           | integer   | peak id number                            |
       +--------------+-----------+-------------------------------------------+
       | specfile     | text      | spectrum file name                        |
       +--------------+-----------+-------------------------------------------+
       | pos          | real      | peak position [nm] (from peak picking)    |
       +--------------+-----------+-------------------------------------------+
       | high         | real      | peak high [arb. u.] (from peak picking)   |
       +--------------+-----------+-------------------------------------------+

    .. note::

       - id is a primary key

    .. table:: peakpar

       +--------------+-----------+-------------------------------------------+
       | column name  | data type | descrip./exmple                           |
       +==============+===========+===========================================+
       | id_peak      | integer   | id of the peak                            |
       +--------------+-----------+-------------------------------------------+
       | parname      | text      | fitted peak param. name (e.g. fwhm_gauss) |
       +--------------+-----------+-------------------------------------------+
       | value        | real      | fitted peak param. value                  |
       +--------------+-----------+-------------------------------------------+
       | error        | real      | fitted peak param. error                  |
       +--------------+-----------+-------------------------------------------+
       | desc         | text      | fitted peak param. description            |
       +--------------+-----------+-------------------------------------------+

    .. note::

       following SQL constraints are applied:
       - id_peak is a foreign key
       - UNIQUE(id_peak, parname) ON CONFLICT REPLACE


    .. table:: assignment

       +---------------+-----------+------------------------------------------+
       | column name   | data type | descrip./exmple                          |
       +===============+===========+==========================================+
       | id_peak       | integer   | id of the peak                           |
       +---------------+-----------+------------------------------------------+
       | id_transition | integer   | id of the transition                     |
       +---------------+-----------+------------------------------------------+
       | prob          | real      | assignment probability                   |
       +---------------+-----------+------------------------------------------+

    .. note::

       following SQL constraints are applied:
       - UNIQUE(id_peak, id_transition) ON CONFLICT REPLACE"""

    def __init__(self):
        self.dbfilen = None
        self.conn = None
        self.cur = None

    def connect(self, dbfilen):
        """Connect the database file."""
        self.close()
        self.dbfilen = dbfilen
        self.conn = sqlite3.connect(dbfilen)
        self.cur = self.conn.cursor()

    def close(self):
        """Close the database connection.

        .. warning::

           this method must be called at the end of the program if a database
           file was connected"""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.dbfilen = None
            self.conn = None
            self.cur = None

    def set_specie(self, name, charge):
        """Set/add a record to the *specie* table.

        Parameters
        ----------
        name : str
            The name of the specie (e.g. "Fe I").
        charge : int
            The charge of the specie (e.g. 0 for Fe I, 1 for Fe II)."""
        self.cur.execute('''INSERT INTO specie (name, charge)
                         VALUES (?,?);''', (name, charge))
        self.conn.commit()

    def set_level(self, id_specie, el_conf, term, j, en, g, ref):
        self.cur.execute('''INSERT INTO level
                         (id_specie, el_conf, term, j, en, g, ref)
                         VALUES (?,?,?,?,?,?,?);''', (id_specie, el_conf,
                                                       term, j, en, g, ref))
        self.conn.commit()

    def set_transition(self, id_specie, id_level_low, id_level_upp, wavelength,
                       aij, ref):
        self.cur.execute('''INSERT INTO transition
                         (id_specie, id_level_low, id_level_upp, wavelength,
                         aij, ref) VALUES (?,?,?,?,?,?);''',
                         (id_specie, id_level_low, id_level_upp, wavelength,
                         aij, ref))
        self.conn.commit()

    def set_peak(self, specfile, pos, high):
        self.cur.execute('''INSERT INTO peak (specfile, pos, high)
                         VALUES (?,?,?);''', (specfile, pos, high))
        self.conn.commit()

    def set_peakpar(self, id_peak, parname, value, error, desc):
        self.cur.execute('''INSERT INTO peakpar
                         (id_peak, parname, value, error, desc)
                         VALUES (?,?,?,?,?);''', (id_peak, parname, value,
                                                  error, desc))
        self.conn.commit()

    def set_assignment(self, id_peak, id_transition, prob):
        self.cur.execute('''INSERT INTO assignment
                         (id_peak, id_transition, prob)
                         VALUES (?,?,?);''', (id_peak, id_transition, prob))
        self.conn.commit()

    def sql_exec(self, sqlstr, pars, fetchall=False, commit=False):
        self.cur.execute(sqlstr, pars)
        if commit:
            self.conn.commit()
        if fetchall:
            return self.cur.fetchall()


def gen_empty_db(dbfilen):
    """Generate an empty database file named *dbfilen*"""
    conn = sqlite3.connect(dbfilen)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE specie
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,     
                charge INTEGER,
                UNIQUE(name, charge) ON CONFLICT IGNORE);''')
    cur.execute('''CREATE TABLE level
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_specie INTEGER,
                el_conf TEXT,
                term TEXT,
                j TEXT,
                en REAL,
                g REAL,
                ref TEXT,
                FOREIGN KEY(id_specie) REFERENCES specie(id),
                UNIQUE(id_specie, el_conf, term, j)
                ON CONFLICT REPLACE);''')
    cur.execute('''CREATE TABLE transition
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_specie INTEGER,
                id_level_low INTEGER,
                id_level_upp INTEGER,
                wavelength REAL,
                aij REAL,
                ref TEXT,
                FOREIGN KEY(id_specie) REFERENCES specie(id),
                FOREIGN KEY(id_level_low) REFERENCES level(id),
                FOREIGN KEY(id_level_upp) REFERENCES level(id),
                UNIQUE(id_specie, id_level_low, id_level_upp)
                ON CONFLICT REPLACE);''')
    cur.execute('''CREATE TABLE peak
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                specfile TEXT,
                pos REAL,
                high REAL);''')
    cur.execute('''CREATE TABLE peakpar
                (id_peak INTEGER,
                parname TEXT,
                value REAL,
                error REAL,
                desc TEXT,
                FOREIGN KEY(id_peak) REFERENCES peak(id),
                UNIQUE(id_peak, parname) ON CONFLICT REPLACE);''')
    cur.execute('''CREATE TABLE assignment
                (id_peak INTEGER,
                id_transition INTEGER,
                prob REAL,
                UNIQUE(id_peak, id_transition) ON CONFLICT REPLACE);''')
    conn.commit()
    conn.close()

