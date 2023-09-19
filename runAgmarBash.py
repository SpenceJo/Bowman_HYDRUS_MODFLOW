#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:18:16 2023

!!! Has all instances simultaneously writing the ATMOSPH.IN input files, which is non-deal

@author: spencerjordan
"""
import os

def runHydrus(numSims):
    groups = {1:'8',
              2:'7',
              3:'6'
                }

    for i in groups:
        wells = groups[i]
        os.system(f"osascript -e 'tell app \"Terminal\" to do script \"cd ~/Documents/Hydrus/python_scripts && conda activate hydrus-modflow && python exec_hydrus_agmar.py <<< {wells} <<< {numSims} \"'")

if __name__ == "__main__":
    print('How many Monte Carlo simulations would you like to run?')
    numSims = int(input(':'))
    runHydrus(numSims)
