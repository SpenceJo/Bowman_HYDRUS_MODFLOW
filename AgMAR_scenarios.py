#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:11:52 2023

Handles some AgMAR scenario analysis

Only includes adjusting original agmar values by some multiplier, and does not
include any new flooding events. 

Goal is to see some 'threshold' of dillution vs pollution

@author: spencerjordan
"""


from hydrus_model_bowman import atmosph,exec_hydrus

def main(nsims,mult):
    a = atmosph(agmar_mult=mult)
    a.main(agmar=True)
    a.write_inputs(agmar=True)
    model = exec_hydrus(nwells=[8],nsims=nsims,cRoot=False)
    model.run_model()

if __name__ == "__main__":
    print('Numer of MC simulations')
    nsims = input(':')
    nsims = int(nsims)
    mults = [0.8,0.6,0.4,0.2]
    for mult in mults:
        main(nsims,mult)
