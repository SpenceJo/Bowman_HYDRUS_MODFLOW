#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:19:05 2023

@author: spencerjordan
"""


from hydrus_model_bowman import exec_hydrus,atmosph
import time
import os

def main(wells,nsims):
    #for mult in [0.2,0.4,0.6,0.8]:
    for mult in [0.15]:
        a = atmosph(agmar_mult=mult)
        a.main(agmar=True)
        a.write_inputs(agmar=True)
        model = exec_hydrus(nwells=wells,nsims=nsims,cRoot=False)
        model.run_model()
        dirName = str(mult)[-1]
        time.sleep(2)
        os.system(f'mv /Users/spencerjordan/Documents/Hydrus/Profiles/result_files/*.p /Users/spencerjordan/Documents/Hydrus/Profiles/result_files/recharge_core_0{dirName}')
        time.sleep(2)

if __name__ == "__main__":
    print('Well numbers as list')
    wells = input(':')
    wells = wells.split(',')
    wells = list(map(int,wells))
    print('Numer of MC simulations')
    nsims = input(':')
    nsims = int(nsims)
    main(wells,nsims)