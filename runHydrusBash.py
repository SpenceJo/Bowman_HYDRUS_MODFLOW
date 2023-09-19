#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:36:36 2023

Run all Hydrus models at once in five different terminals
Loads CPU to about 80% and eats up a ton of RAM

@author: spencerjordan
"""

import os

def runHydrus(numSims,AgMAR):
    if AgMAR == 'yes':
        groups = {1:'8',
                  2:'7',
                  3:'6'
                    }
    else:
        groups = {
                  1:'1,2,3,4',
                  2:'5,6,7,8',
                  3:'9,10,11,12',
                  4:'13,14,15,16',
                  5:'17,18,19,20'
                 }
    for i in groups:
        wells = groups[i]
        os.system(f"osascript -e 'tell app \"Terminal\" to do script \"cd ~/Documents/Hydrus/python_scripts && conda activate hydrus-modflow && python exec_hydrus.py <<< {wells} <<< {numSims} \"'")

if __name__ == "__main__":
    print('How many Monte Carlo simulations would you like to run?')
    numSims = int(input(':'))
    print('Are you running AgMAR?')
    AgMAR = input(':')
    runHydrus(numSims,AgMAR)
