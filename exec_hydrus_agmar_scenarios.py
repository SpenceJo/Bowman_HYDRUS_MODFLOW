#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:56:50 2023

@author: spencerjordan
"""

from hydrus_model_bowman import atmosph,exec_hydrus

def main(wells,nsims):
    for mult in [0.2,0.4,0.6,0.8]:
        a = atmosph(agmar_mult=mult)
        a.main(agmar=True)
        a.write_inputs(agmar=True)
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