#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:37:53 2023

Class to run hydrus and save outputs
Want to keep it seperated so that we can run 5 instances of it at once
Organized such as to be run from runHydrusBash.py, instead of running all at once in serial

@author: spencerjordan
"""

from hydrus_model_bowman import exec_hydrus

def main(wells,nsims):
    model = exec_hydrus(nwells=wells,nsims=nsims,cRoot=False)
    model.run_model()

if __name__ == "__main__":
    print('Well numbers as list')
    wells = input(':')
    wells = wells.split(',')
    wells = list(map(int,wells))
    print('Numer of MC simulations')
    nsims = input(':')
    nsims = int(nsims)
    main(wells,nsims)
















