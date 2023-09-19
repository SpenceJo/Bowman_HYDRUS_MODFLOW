#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:34:29 2023

Class to control Bowman Modflow/MT3DMS Model

Trying to make things more modular

Running an AgMAR flooding at the original 05/03/2022 date as well as every January
1st in 10-year intervals afterwards. SPs for AgMAR are always 59 days, but we can
set the flooding amounts to whatever we want in the Hydrus model, so we do not 
always need identical flooding amounts, this could be used to account for dry years

@author: spencerjordan
"""

import flopy 
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
from tqdm import tqdm
import json
import pylab as pl
import numpy.lib.recfunctions as rfn


## Creating a class for the flopy model object, a class within a class??
class modflow(object):
    """
    Bowman MODFLOW-2005 Model
    """
    def __init__(self):
        ## Path to main working directory
        self.workingDir = '/Users/spencerjordan/Documents/modflow_work'
        ## Name of the model
        self.modelname = 'Bowman_Modflow_Recharge'
        ## Path to model directory
        self.model_ws = '/Users/spencerjordan/Documents/modflow_work/recharge_model/'
        ## Start date of the model
        self.startDate = pd.to_datetime('1987-09-01')
        ## End date of the model
        self.endDate = pd.to_datetime('2099-08-31')
        ## Date of first AgMAR event in May of 2023
        self.agmarDate = pd.to_datetime('2022-05-03')
        ## Interval of scheduled AgMAR floodings
        self.AgMAR_interval = 10
    
    
    ########################################################
    ## Functions to handle pickling/unpickling model objects
    ########################################################
    def object_save(self,obj=object,name=str):      
        file_to_store = open(name+'.pickle', 'wb')
        pickle.dump(obj, file_to_store)
        file_to_store.close()
        
    def object_load(self,name=str):
        file_to_read = open(name+'.pickle', "rb")
        obj = pickle.load(file_to_read)
        file_to_read.close()
        return obj
    
    
    ###########################
    ## Define the Modflow model
    ###########################
    def defineModel(self):
        self.mf = flopy.modflow.Modflow(self.modelname, exe_name='/Users/spencerjordan/Documents/pymake_modflow/examples/mf2005', model_ws=self.model_ws)
        #mf = flopy.modflow.Modflow.load(model_ws+modelname+'.nam')
        self.mf.exe_name = '/Users/spencerjordan/Documents/pymake_modflow/examples/mf2005'


    ##############################
    ## Define model discretization
    ##############################
    def descrit(self):
        Lx = 1092 # --> I feel like this is actually 1500, 12*125 = 1500...
        Ly = 1404
        self.nlay = 125
        self.nrow = 117
        self.ncol = 91
        #####################################
        ## Calculate number of stress periods
        #####################################
        ## First, calculate number of SP's without including AgMAR --> Should be equal to 672 for original model
        nper = int(round((self.endDate - self.startDate)/np.timedelta64(2, 'M')))
        ## Next, calculate the number of SP's required for AgMAR
        nperAgmar = 59 # 59 days for ACTUAL flooding in May 2022
        ## Add an additional 59 SP's for every Xth year Winter (January 1st) following that 
        ## Date of first additional flooding
        firstFlood = pd.to_datetime(f'20{22+self.AgMAR_interval}-01-01')
        ## Number of floodings after the original
        self.numAgmar = int(np.floor((self.endDate - firstFlood)/np.timedelta64(self.AgMAR_interval, 'Y'))) + 1
        nperAgmar += 59 * self.numAgmar
        ## Total numbner of SPs = oringal number + agmar SPs - number of agmar SPs (since they replace an original 2-month SP)
        ## Subtracting one more to account for the replacement of the original flooding
        nper = nper + nperAgmar - self.numAgmar - 1
        self.nper = nper
        ## Spacings along a row
        delr = 12 
        ## Spacings along a column
        delc = 12
        self.delz = 0.32
        ###########################################################################
        ## Assign SP length, all AgMAR SP's are daily and all others are bi-monthly.
        ## Could be some errors in here, tested with 5, 7, and 10 year intervals
        ## and that worked okay
        ###########################################################################
        SP_lengthNorm = [2*30.438]
        SP_lengthAgmar = [1]
        ## Number of SP's pre AgMAR --> all bimonthly
        self.preAgmarNum = int(round((self.agmarDate - self.startDate)/np.timedelta64(2, 'M')))
        ## Numer of SP's per AgMAR flooding --> 59 daily SP's
        self.agmarNum = 59
        ## Numer of bimonthly SP's between original flooding and first January flooding
        self.interNumFirst = int(round((firstFlood - self.agmarDate)/np.timedelta64(2,'M')) - 1)
        ## Number of bimonthly SP's between the repeated floodings
        self.interNum = int(round((pd.to_datetime(f'20{22+(self.AgMAR_interval*2)}-01-01') - firstFlood)/np.timedelta64(2,'M')) - 1)
        ## Number of bimonthly SP's between last flooding and end of simulation
        ## Date of last flooding
        lastFlood = pd.to_datetime(f'20{22+self.AgMAR_interval*self.numAgmar}-01-01')
        ## Number of SPs after last flooding
        self.interNumEnd = int(round((self.endDate - lastFlood)/np.timedelta64(2, 'M') - 1))
        ## Start with preAgmar, first flooding, and first break between floods
        perlen = (SP_lengthNorm * self.preAgmarNum) + (SP_lengthAgmar * self.agmarNum) + (SP_lengthNorm * self.interNumFirst)
        ## Now add a flooding and a break for each Agmar event 
        for i in range(self.numAgmar):
            perlen = perlen + (SP_lengthAgmar * self.agmarNum)
            if i != (self.numAgmar-1):
                perlen = perlen + (SP_lengthNorm * self.interNum)
            ## After last flooding, only add on the number of SPs remaining
            else:
                perlen = perlen + (SP_lengthNorm * self.interNumEnd)
        self.perlen = perlen
        ###################################################
        ## Do the same as above for the number of timesteps
        ###################################################
        nstp = ([2] * self.preAgmarNum) + ([3] * self.agmarNum) + ([2] * self.interNumFirst)
        for i in range(self.numAgmar):
            nstp = nstp + ([3] * self.agmarNum)
            if i != (self.numAgmar-1):
                nstp = nstp + ([2] * self.interNum)
            ## After last flooding, only add on the number of SPs remaining
            else:
                nstp = nstp + ([2] * self.interNumEnd)
        ## Timestep multiplier
        tsmult = 1 
        ## Steady state true or false --> False for all stress periods == Transient model
        steady = np.zeros(self.nper, dtype=bool)   
        ## Time units (days)
        itmuni = 4
        ## Length units (meters)
        lenuni = 2
        ## X and Y coords of upper left corner of the grid
        xul = 1947413.652
        yul = 624770.691 + Ly
        ## Projection information
        rotation = 0
        proj4_str = 'epsg:26943'
        start_datetime = '1987-09-01'
        ## Load the spatial data for the top and bottom of the model
        self.Top = np.loadtxt('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/Top_ele_lay1.dat',dtype='f', delimiter= None)
        Bottom = np.loadtxt('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/Bottom_ele.dat', dtype='f', delimiter=None)
        ## Reshape the bottom layer
        Bottom = Bottom.reshape((self.nlay,self.nrow,self.ncol))
        self.Bottom = Bottom
        #Check to make sure bottom data interpolated correctly
        #plt.imshow(Bottom[1,:,:], interpolation=None)
        ## Define ModflowDis package
        flopy.modflow.ModflowDis(model = self.mf,
                                 nlay = self.nlay,
                                 nrow = self.nrow, 
                                 ncol = self.ncol,
                                 nper = self.nper,
                                 delr = delr,
                                 delc = delc,
                                 top = self.Top,
                                 botm = Bottom,
                                 perlen = perlen,
                                 nstp = nstp,
                                 tsmult = tsmult,
                                 steady = steady,
                                 itmuni = itmuni,
                                 lenuni = lenuni,
                                 xul = xul,
                                 yul = yul,
                                 rotation = rotation,
                                 proj4_str = proj4_str,
                                 start_datetime = start_datetime)
        
        
    ###############################
    ## Define modflow basic package
    ###############################
    def basic(self):
        print('Loading Modlfow Basic Package')
        ibound = np.ones((self.nlay, self.nrow, self.ncol), dtype=np.int32)
        # ibound[9:-1, :, 0] = -1 # W boundary. Bottom is WEL package
        # ibound[9:-1, :, -1] = -1 # E boundary
        # ibound[9:-1, 0, :] = -1 # N boundary
        # ibound[9:-1, -1, :] = -1 # S boundary
        
        ## Use heads from a previous solve 
        #strt = np.load("Initial_heads.npy")[1,:,:]
        ## Starting head around layer 10 (1 default; doesn't work well)
        strt = self.Top - 3.2
        ifrefm = True # free format T/F
        hnoflo = 999 # no flow head value
        flopy.modflow.ModflowBas(self.mf, 
                                 ibound=ibound,
                                 strt=strt,
                                 ifrefm=ifrefm,
                                 hnoflo=hnoflo)
       
        
    def layer_property(self):
        print('Loading Layer Properties')
        ## Save cell by cell flows is >1
        ipakcb = 1
        ## <0 confined. >0 convertible.
        laytyp = np.ones(self.nlay, dtype='int') 
        ## Interblock conductivity: 0 is harmonic mean. 
        layavg = np.zeros(self.nlay, dtype='int')
        ## Horizontal anisotropy. None is 1.
        chani = np.ones(self.nlay, dtype='int')
        ## VKA is vertical hydraulic conductivity (not a multiplier)
        layvka = np.zeros(self.nlay, dtype='int')
        ## Wetting active >0
        laywet = np.ones(self.nlay, dtype='int')
        ## Factor converting dry to wet.
        wetfct = 0.5
        iwetit = 5 # 10 #rewetting interval
        ## Determines equation to define initial head in wetted cells; 0 relies on head in neighboring cells and bottom cells.
        ihdwet = 1 # 0
        ## Rewetting from bottom only: default is -.01
        iwetdry = -0.9
        hdry = -999 
          
        ## Import zones from TPROGS to assign parameters based on zones
        tsim_file = '/Users/spencerjordan/Documents/modflow_work/Geology/tsim_BowmanN_1404x1092.asc1'
        tsim = np.loadtxt(tsim_file,dtype='int', skiprows=1)
        ## nlay, nrow, ncol
        tsim = tsim.reshape((125,117,91))
        ## origin of tsim is bottom corner (-20m); layer 125, need to flip upside down
        tsim = np.flip(tsim,axis=[0,1])
        tsim = abs(tsim)
        ###################
        ## Parameter values
        ###################
        param_val = [1.61761034e+02, 4.27012155e+01, 8.08407921e-01, 8.58018938e-02,
                     3.02550754e-01, 3.42047640e-01, 2.59482294e-01, 3.94801531e-01,
                     2.08937899e-01, 2.31413266e-01, 1.30768420e-01, 1.25146707e-02]
        ## Choose a set of hydraulic parameters (one for each layer, four total)
        HK = param_val[0:4]
        ## Vertical K is 1/10 that of horizontal
        VK = [HK[0]/10, HK[1]/10, HK[2]/10, HK[3]/10]
        ## Specific yield
        Sy = param_val[8:12]
        ## Speciic storage
        Ss = [7.53E-6, 1.04E-5, 1.36E-5, 1.47E-5]
        
        ## Set of parameters set by Hanni at some point
        # HK = [332.6654711, 103.2874366, 1.678084206, 0.11170148]
        # VK = [HK[0]/10, HK[1]/10, HK[2]/10, HK[3]/10]
        # Ss = [7.53E-6, 1.04E-5, 1.36E-5, 1.47E-5]
        # Sy = [0.27, 0.21, 0.18, 0.094]
        
        hk = np.ones((self.nlay,self.nrow,self.ncol), dtype='f')
        hani = np.ones((self.nlay,self.nrow,self.ncol), dtype=np.int32)
        vk = np.ones((self.nlay,self.nrow,self.ncol), dtype='f')
        ss = np.ones((self.nlay,self.nrow,self.ncol), dtype='f')
        sy = np.ones((self.nlay,self.nrow,self.ncol), dtype='f')
        
        ## Assign values to match the soil types described by the tsim realization
        for i in range(0,4):
            ## finding the index for each soil type and setting respective parameter
            hk[tsim==(i+1)] = HK[i]
            vk[tsim==(i+1)] = VK[i]
            ss[tsim==(i+1)] = Ss[i]
            sy[tsim==(i+1)] = Sy[i]    
        
        flopy.modflow.ModflowLpf(model=self.mf,
                                 ipakcb = ipakcb,
                                 laytyp = laytyp, 
                                 layavg = layavg, 
                                 chani = chani, 
                                 layvka = layvka, 
                                 laywet = laywet, 
                                 wetfct = wetfct, 
                                 iwetit = iwetit, 
                                 ihdwet = ihdwet,
                                 wetdry = iwetdry,
                                 hdry = hdry,
                                 hk = hk, 
                                 hani = hani, 
                                 vka = vk, 
                                 ss = ss, 
                                 sy = sy)
            
        
    def recharge(self):
        print('Assigning Recharge Data')
        ###############################################################
        ## Load Hydrus results that do not include the daily AgMAR data
        ###############################################################
        with open('/Users/spencerjordan/Documents/Hydrus/python_scripts/agmar_pickle_files/recharge.p','rb') as handle:
            recharge = pickle.load(handle)
        ## Convert cm of recharge into m of recharge
        recharge['recharge'] /= 100
        ## Prepare sp column to be grouped bi-monthly
        recharge.sp = (recharge.sp/2).round(0).astype(int)
        recharge.index = recharge.index.astype(int)
        ## Convert everything into numeric data
        recharge['sp'] = pd.to_numeric(recharge['sp'])
        recharge['row'] = pd.to_numeric(recharge['row'])
        recharge['col'] = pd.to_numeric(recharge['col'])
        recharge['recharge'] = pd.to_numeric(recharge['recharge'])
        ## Groupby stress period, row, and column to get average values for the stress periods
        ## for each cell in the modflow grid
        recharge = recharge.groupby(by=['sp','row','col'], as_index=False).mean()
        ## The recharge data is setup as a grid that is indexed at 1, change to be zero indexed
        recharge.row = recharge.row - 1
        recharge.col = recharge.col - 1
        ##!!! No negative recharge; maybe reconsider this?
        recharge.loc[recharge['recharge'] < 0,'recharge'] = 0
        
        #######################################################################
        ## Edit the SP data to prepare for the inclusion of the AgMAR floodings
        #######################################################################
        ## First stress period to replace with AgMAR data
        start = 208
        for x in range(self.numAgmar+1):
            recharge.loc[recharge['sp']>start,'sp'] = recharge.loc[recharge['sp']>start,'sp'] + 58
            recharge = recharge[recharge['sp'] != start]
            ## Number of SPs until next flooding
            if x == 0:
                start += self.interNumFirst + 58
            else:
                start += self.interNum + 58
                
        ##################################################
        ## Read in the AgMAR recharge datasets from HYDRUS
        ##################################################
        ## Loop through for each flooding, +1 added becuase numAgmar is number of floodings after May 2022 flood
        start = 208
        for i in range(self.numAgmar+1):
            print(f'applying agmar {i}')
            ## Load the AgMAR recharge data for each flooding
            with open(f'/Users/spencerjordan/Documents/Hydrus/python_scripts/agmar_pickle_files/recharge_agmar_{i}.p','rb') as handle:
                recharge_rch = pickle.load(handle)
            ## Convert cm of recharge into m of recharge
            recharge_rch['recharge'] /= 100
            ## Convert everything into numeric data
            recharge_rch.index = recharge_rch.index.astype(int)
            recharge_rch['sp'] = pd.to_numeric(recharge_rch['sp'])
            recharge_rch['row'] = pd.to_numeric(recharge_rch['row'])
            recharge_rch['col'] = pd.to_numeric(recharge_rch['col'])
            recharge_rch['recharge'] = pd.to_numeric(recharge_rch['recharge'])
            ## Groupby stress period, row, and column to get average values for the stress periods
            ## for each cell in the modflow grid
            recharge_rch = recharge_rch.groupby(by=['sp','row','col'], as_index=False).mean()
            ## The recharge data is setup as a grid that is indexed at 1, this changes it to be zero indexed
            recharge_rch.row = recharge_rch.row-1
            recharge_rch.col = recharge_rch.col-1    
            ##!!! Don't want negative recharge
            recharge_rch.loc[recharge_rch.recharge<0,'recharge'] = 0
            ## Edit to SP's to match that of the original model
            if i == 0:
                recharge_rch['sp'] = recharge_rch['sp'] + 207
            elif i == 1:
                recharge_rch['sp'] = recharge_rch['sp'] + 207 + 58 + self.interNumFirst
            else:
                recharge_rch['sp'] = recharge_rch['sp'] + 207 + (i * 58) + self.interNumFirst + (self.interNum * (i - 1))
            ## Merge the original and AgMAR datasets together
            recharge_1 = recharge[recharge.sp<start]
            recharge_2 = recharge[recharge.sp>start]
            recharge = pd.concat([recharge_1,recharge_rch,recharge_2])
            ## Number of SPs until next flooding
            if x == 0:
                start += self.interNumFirst + 58
            else:
                start += self.interNum + 58
        recharge = recharge.reset_index(drop=True)
        ## Create dataset in format Flopy requires
        rech = dict()
        for i in tqdm(recharge.sp.unique()):
            sub = recharge[recharge.sp==i]
            col = np.array(sub.col)
            row = np.array(sub.row)
            data = np.array(sub.recharge)
            grid = np.zeros((self.mf.nrow,self.mf.ncol))
            grid[col,row] = data
            rech[i] = grid
        ## Do not save cell by cell
        ipakcb = 0
        ## Apply recharge to highest active cell
        nrchop = 3
        flopy.modflow.ModflowRch(model=self.mf,
                                 nrchop=nrchop,
                                 ipakcb=ipakcb,
                                 rech=rech)
        self.rech = rech


    def rechargeContours(self):
        try:
            for sp in self.rech:
                ## Recharge plotting contours
                X, Y = np.mgrid[0:1:117j, 0:1:91j]
                fig, ax = pl.subplots(figsize=(15, 10))
                c1 = ax.contourf(X, Y, self.rech[sp])
                ax.set_title(f'Stress Period: {sp}',fontsize=20)
                pl.colorbar(c1, ax=ax);
                plt.show()
        except:
            print('**** Please run rechargeFunc before plotting ****')
            
        
    def wel(self,rch):
        print('Assigning Well Data')
        ## Do not save cell by cell
        ipakcb = 3
        ## Well pumping plus bottom flux = 11,392, reduce proportionally by area
        targetflux = -11392*(1404*1092)/(2800*2800)
        
        # Adjust bottom/well flux proportionally to annual recharge -> proportional to recharge in regional model of 70cm/yr
        # First find annual recharge rate for 30 years stress period data
        rch_sp = np.zeros(self.nper)
        print('Adjusting Bottom Flux')
        for i in range(0,self.nper):
            rch_sp[i] = self.rech[i].sum()/(self.nrow*self.ncol) #depth recharge per day to compare to daily bottom flux in prior model
            #rch_sp[i] = rech[i].mean()
        # plt.plot(rch_sp) #inspect recharge per stress period
        
        #Try doing assessing recharge/flux per sp not per year in units of cm/day *m/d
        sp = np.arange(0,self.nper)
        rch_df = pd.DataFrame(data={'recharge': rch_sp, #m depth
                                    'sp': sp},
                              index = sp)
        #Calculate multiplier compared to 70 cm/yr converted to cm/d *m/d
        rch_df['flux'] = np.ones(self.nper)*targetflux/(self.nrow*self.ncol)
        rch_df['multiplier'] = rch_df.recharge/(.70/365.25) 
        
        rch_df['flux_adj'] = rch_df.flux * rch_df.multiplier
        rch_df.flux_adj.plot()
        print('Saving adjusted flux data')
        rch_df.flux_adj.to_csv(f'{self.workingDir}/flux_adj.csv', index=False)
        flux_adjusted = rch_df
        
        #stress_period_data: Dictionary w/ lists of [lay, row, col, flux]. 
        # Indices of dictionary is stress period.
        stress_period_data = dict() #One dictionary for each stress period
        print('Creating Stress Period Data')
        for k in range(0,self.nper):
            spd = list()
            for i in range(0,self.nrow):
                for j in range(0,self.ncol):
                    spd.append([self.nlay-1,i,j,rch_df.flux_adj[k]])
            stress_period_data[k] = spd
        
        flopy.modflow.ModflowWel(self.mf,
                                 ipakcb,
                                 stress_period_data)
        ## Average vertical 
        kavg = stats.hmean(np.stack(self.mf.lpf.vka.get_value()).flatten())
        #kavg = stats.hmean(vk[vk>VK[2]])
        Q = flux_adjusted.flux_adj * self.nrow * self.ncol 
        A = 1404 * 1092
        #A = 1404*1092*0.33 #flow is only through coarse fraction (33% coarse)
        v_grad = -Q / (A * kavg)
        ##!!! Talk to Thomas or Leland Here
        v_grad[210:268] = np.mean(v_grad[0:214]) 
        ## Save outputs
        self.v_grad = v_grad
        self.rch_df = rch_df


    def load_chd_data(self):
        """
        Load and prep boundary head data for CHD package
        """
        ## Load interpolated boundary heads
        head_data = pd.read_csv('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/Interp_boundary_heads2m.csv')
        data = head_data.copy()
        #############################################################################
        ## Create CHD dataset as if no flooding has occured, similar to recharge data
        ## Method is similar to what is used in the recharge function
        #############################################################################
        ID = data.ID.to_numpy()
        ID_all = ID
        ## Length of SP's --> 2-months
        perlen = max(self.perlen)
        ########################################################################
        ## Repeat the data onto itself for the number of years in the simulation
        ## 672 is the length of the simulation WITHOUT AgMAR, assuming 2-month 
        ## stress periods
        ########################################################################
        for i in range(0,(int(672*perlen/365.25)-1)):
            ID_all = np.concatenate([ID_all,ID+(6*(i+1))]) #6 is number of sp per year
        ## Agian, repeat for number of simulation years
        data = pd.concat([data]*int(672*perlen/365.25)) 
        data['sp'] = ID_all-1 #zero index
        ###################################################
        ## Prep data to be ready to receive AgMAR floodings
        ###################################################
        start = 208
        for x in range(self.numAgmar+1):
            data.loc[data['sp']>start,'sp'] = data.loc[data['sp']>start,'sp'] + 58
            data = data[data['sp'] != start]
            ## Number of SPs until next flooding
            if x == 0:
                start += self.interNumFirst + 58
            else:
                start += self.interNum + 58
        ########################################
        ## Assign the data to AgMAR time periods
        ########################################
        agmarHds_1 = head_data[head_data['ID']==4]
        agmarHds_2 = head_data[head_data['ID']==2]
        nextSP_1 = head_data[head_data['ID']==4]
        nextSP_2 = head_data[head_data['ID']==2]
        for i in range(59):
            nextSP_1['ID'] = nextSP_1['ID'] + 1
            nextSP_2['ID'] = nextSP_2['ID'] + 1
            agmarHds_1 = pd.concat([agmarHds_1,nextSP_1])
            agmarHds_2 = pd.concat([agmarHds_2,nextSP_2])
        agmarHds_1['ID'] = agmarHds_1['ID'] - 4
        agmarHds_2['ID'] = agmarHds_2['ID'] - 2
        agmarHds_1['sp'] = agmarHds_1['ID']
        agmarHds_2['sp'] = agmarHds_2['ID']
        start = 208
        for i in range(self.numAgmar+1):
            ## Edit to SP's to match that of the original model
            if i == 0:
                agmarHds = agmarHds_1.copy()
                agmarHds['sp'] = agmarHds['sp'] + 207
                data_1 = data[data.sp<start]
                data_2 = data[data.sp>start]
                data = pd.concat([data_1,agmarHds,data_2])
            elif i == 1:
                agmarHds = agmarHds_2.copy()
                agmarHds['sp'] = agmarHds['sp'] + 207 + 58 + self.interNumFirst
                data_1 = data[data.sp<start]
                data_2 = data[data.sp>start]
                data = pd.concat([data_1,agmarHds,data_2])
            else:
                agmarHds = agmarHds_2.copy()
                agmarHds['sp'] = agmarHds['sp'] + 207 + (i * 58) + self.interNumFirst + (self.interNum * (i - 1))
                data_1 = data[data.sp<start]
                data_2 = data[data.sp>start]
                data = pd.concat([data_1,agmarHds,data_2])
            ## Number of SPs until next flooding
            if x == 0:
                start += self.interNumFirst + 58
            else:
                start += self.interNum + 58    
        data = data.reset_index(drop=True)
        data['row'] = data.row-1
        data['col'] = data.col-1
        self.chd_data = data


    def chd(self,v_grad,rch_df):
        print('Initializing CHD Package....')
        print("Is chd data already JSON'd? (yes/no)")
        chd_pickle = input(':')
        if chd_pickle =='no':
            data = self.chd_data
            
            #Annual average situation applied - need factor conversion for times with lower water level
            #multiplier = pd.read_csv('C:/Users/rache/Desktop/GWV simple model/GWVplay/BowmanN/1404x1092/chd_wl_factor_021622.csv')
            multiplier = pd.read_csv('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/regional_wldata2m_3.csv',index_col=0,parse_dates=True)
            
            start_date = '1987-09-01'
            ## Changed from 2047
            end_date = '2099-09-01'
            index = pd.date_range(start=start_date,end=end_date,freq='2M') #Try 2 month time steps??? 2M
            multiplier = multiplier[multiplier.index.isin(index)] #keep only times in model currently
            # Need to add predicted multiplier for future (2021-2047) --> 2021-2100 Spencer 
            start_date = '2021-09-01'
            end_date = '2099-09-01' #'2018-10-31'
            index = pd.date_range(start = start_date, end = end_date, freq='2M') #Try 2 month time steps??? 2M
            
            # Repeat factor between Sept 1 2013 - Sept 1 2021 because that is what climate is modeled as... (not great but not many other options)
            future_data = pd.concat([multiplier.loc['2013-09-01':'2021-09-01']]*10).iloc[0:len(index)-1,:]
            future_data.index = index[1::]
            
            #Combine back together
            multiplier = pd.concat([multiplier[multiplier.index<'2021-10-01'], future_data])
            
            MEAN_WT = 12.864807804386265 #from site data since that is used to set chds
            multiplier['region_factor'] = multiplier['WL'] - MEAN_WT
            
            
            ###################################################################
            ########## Adjusting multiplier to fit AgMAR scenarios ############
            ###################################################################
            ##!!! Quick fix to get 10-year model running, make more dynamic later
            ## ^^ To do this, need to just change some dates below to be based on the agmar interval
            startDays = [pd.to_datetime('05/31/2022',format='%m/%d/%Y')]
            ## Repeat flooding every 10th Winter on January 1st
            for i in range(7):
                startDays.append(pd.to_datetime(f'01/31/20{32+i*10}',format='%m/%d/%Y'))
            ## Copy the multiplier for each relevant 2-month SP to the 59 daily SP's during recharge
            for j,floodDate in enumerate(startDays):
                if j == 0:
                    wl = [multiplier['WL'][self.preAgmarNum]]*58
                    rf = [multiplier['region_factor'][self.preAgmarNum]]*58
                    mult_rch_idx = pd.date_range('2022-05-02 00:00:00','2022-06-28 00:00:00', freq='1D')
                elif j == 1:
                    wl = [multiplier['WL'][self.preAgmarNum+58+self.interNumFirst]]*58
                    rf = [multiplier['region_factor'][self.preAgmarNum+58+self.interNumFirst]]*58
                    mult_rch_idx = pd.date_range('2032-01-01 00:00:00','2032-02-27 00:00:00', freq='1D')
                else:
                    wl = [multiplier['WL'][self.preAgmarNum + (58 * j) + self.interNumFirst + (self.interNum * (j-1))]] * 58
                    rf = [multiplier['region_factor'][self.preAgmarNum + (58 * j) + self.interNumFirst + (self.interNum * (j-1))]] * 58
                    mult_rch_idx = pd.date_range(f'20{32+(self.AgMAR_interval * (j-1))}-01-01 00:00:00',f'20{32+(self.AgMAR_interval * (j-1))}-02-27 00:00:00', freq='1D')
                mult_rch = pd.DataFrame({'WL':wl,'region_factor':rf})
                mult_1 = multiplier[multiplier.index<floodDate]
                ## This is a bit fuzzy, I feel like I'm one SP off
                mult_2 = multiplier[multiplier.index>=floodDate]
                mult_rch = mult_rch.set_index(mult_rch_idx)
                
                ## Create the recharge multiplier dataframe
                multiplier_rch = pd.concat([mult_1,mult_rch,mult_2])
                multiplier = multiplier_rch
            self.multiplier = multiplier
            ###################################################################
            ###################################################################
            
            multiplier['rch_factor'] =  (self.rch_df.recharge.values/(0.2/365))
            # savethis = multiplier.region_factor + multiplier.rch_factor
            # multiplier.to_csv('recharge_factor.csv')
            
            #Combine recharge and regional factors
            C = 0.035
            R = 0.7
            k = 0.7
            multiplier['adj_factor'] = C*multiplier.rch_factor + R*multiplier.region_factor + k
            multiplier['factor'] = multiplier.adj_factor
           
            # multiplier.to_csv('recharge_factor.csv')
            print('Applying multiplier') 
            for i in range(0,self.nper):
                sub = data[data.sp==i]
                new_value = sub['boundary_heads'] + multiplier['factor'][i]   #try adjusting by subtracting  
                data.loc[data['sp']==i,'boundary_heads'] = new_value
            
            data['wt_lay'] = abs(self.Bottom[:,data.row,data.col]-np.tile(data.boundary_heads,(125,1))).argmin(axis=0)
            print('Saving CHD Data to csv')
            data.to_csv(f'{self.workingDir}/chd_data.csv')
            
            Eslice = np.zeros((self.nper,self.nlay,self.nrow))
            Wslice = np.zeros((self.nper,self.nlay,self.nrow))
            Nslice = np.zeros((self.nper,self.nlay,self.ncol))
            Sslice = np.zeros((self.nper,self.nlay,self.ncol))
            print('Writing water table data')
            for i in tqdm(range(0,data.sp.max())): #I have 12 months of data I will repeat: September through September
                sub = data[data.sp==i]
                
                E = sub[sub['col']==(self.ncol-1)]
                W = sub[sub['col']==0]
                N = sub[sub['row']==0]
                S = sub[sub['row']==(self.nrow-1)]
                
                #match the water table to the right layer like I did in R 
                #start at wt_lay and decrease by v_grad until wt_lay+count==nlay-1
                #Then start at wt_lay and increase by v_grad until wt_lay+count==0
                #Eslice
                for j in range(0,self.nrow):
                    count = 0
                    wt_lay = E.wt_lay.iloc[j]
                    while(wt_lay+count<=(self.nlay-1)):
                        Eslice[i,wt_lay+count,j] = E.boundary_heads.iloc[j]-v_grad[i]*count*self.delz
                        count += 1
                    
                    count = 0
                    while(wt_lay-count>=0):
                        Eslice[i,wt_lay-count,j] = E.boundary_heads.iloc[j]+v_grad[i]*count*self.delz
                        count += 1
                
                #Wslice
                for j in range(0,self.nrow):
                    count = 0
                    wt_lay = W.wt_lay.iloc[j]
                    while(wt_lay+count<=(self.nlay-1)):
                        Wslice[i,wt_lay+count,j] = W.boundary_heads.iloc[j]-v_grad[i]*count*self.delz
                        count += 1
                    
                    count = 0
                    while(wt_lay-count>=0):
                        Wslice[i,wt_lay-count,j] = W.boundary_heads.iloc[j]+v_grad[i]*count*self.delz
                        count += 1
                
                #Nslice
                for j in range(0,self.ncol):
                    count = 0
                    wt_lay = N.wt_lay.iloc[j]
                    while(wt_lay+count<=(self.nlay-1)):
                        Nslice[i,wt_lay+count,j] = N.boundary_heads.iloc[j]-v_grad[i]*count*self.delz
                        count += 1
                    
                    count = 0
                    while(wt_lay-count>=0):
                        Nslice[i,wt_lay-count,j] = N.boundary_heads.iloc[j]+v_grad[i]*count*self.delz
                        count += 1
                
                #Sslice
                for j in range(0,self.ncol):
                    count = 0
                    wt_lay = S.wt_lay.iloc[j]
                    while(wt_lay+count<=(self.nlay-1)):
                        Sslice[i,wt_lay+count,j] = S.boundary_heads.iloc[j]-v_grad[i]*count*self.delz
                        count += 1
                    
                    count = 0
                    while(wt_lay-count>=0):
                        Sslice[i,wt_lay-count,j] = S.boundary_heads.iloc[j]+v_grad[i]*count*self.delz
                        count += 1
            
            print('Grouping Data by Stress Period')
            stress_period_data = {} #One dictionary for each stress period# lay row col stage cond
            for k in tqdm(range(0,self.nper)):
                spd = list()
                
                # sub = data[data.sp==k]
                
                # E = sub[sub['col']==(ncol-1)]
                # W = sub[sub['col']==0]
                # N = sub[sub['row']==0]
                # S = sub[sub['row']==(nrow-1)]
                
                # chd_start_lay = max([E.wt_lay.max(), W.wt_lay.max(), N.wt_lay.max(), S.wt_lay.max()])
                # print(k,chd_start_lay)
                
                if(k==0): #Start and end head are same
                    for i in range(30,self.nlay-1):
                        for j in range(0,self.nrow):
                            # E
                            if (Eslice[k,i,j]!=0):
                                spd.append([i,j,self.ncol-1,Eslice[k,i,j],Eslice[k,i,j]])  
                        for j in range(0,self.nrow):
                            # W
                            if (Wslice[k,i,j]!=0):
                                spd.append([i,j,0,Wslice[k,i,j],Wslice[k,i,j]])
                        
                        for j in range(1,self.ncol-1):
                            # N
                            if (Nslice[k,i,j]!=0):
                                spd.append([i,0,j,Nslice[k,i,j],Nslice[k,i,j]])
                        for j in range(1,self.ncol-1):
                            # S
                            if (Sslice[k,i,j]!=0):
                                spd.append([i,self.nrow-1,j,Sslice[k,i,j],Sslice[k,i,j]])
                
                if(k>0): #start head matches previous end head
                    for i in range(30,self.nlay-1):
                        for j in range(0,self.nrow):
                            # E
                            if (Eslice[k,i,j]!=0):
                                spd.append([i,j,self.ncol-1,Eslice[k-1,i,j],Eslice[k,i,j]])  
                        for j in range(0,self.nrow):
                            # W
                            if (Wslice[k,i,j]!=0):
                                spd.append([i,j,0,Wslice[k-1,i,j],Wslice[k,i,j]])
                        for j in range(1,self.ncol-1):
                            # N
                            if (Nslice[k,i,j]!=0):
                                spd.append([i,0,j,Nslice[k-1,i,j],Nslice[k,i,j]])
                        for j in range(1,self.ncol-1):
                            # S
                            if (Sslice[k,i,j]!=0):
                                spd.append([i,self.nrow-1,j,Sslice[k-1,i,j],Sslice[k,i,j]])
                stress_period_data[k] = spd
            
            ## For some reason there's no data for last SP. Repeat previous to last
            stress_period_data[1135] = stress_period_data[1134]
            ## Save stress_period_data dictionary as a .json file
            with open("/Users/spencerjordan/Documents/modflow_work/python_scripts/sp_dict_multi_rch.json", "w") as outfile:
                json.dump(stress_period_data, outfile)
        
        else:
            ###########################################################################
            ############ Start here if JSON file has already been created #############
            ###########################################################################
            with open("/Users/spencerjordan/Documents/modflow_work/python_scripts/sp_dict_multi_rch.json", "r") as infile:
                stress_period_data = json.load(infile)
        
        self.stress_period_data = stress_period_data
        print('Creating and Writing CHD File')
        flopy.modflow.ModflowChd(self.mf, 
                                 stress_period_data)
        #chd.write_file()


    def outputControl(self):
        print('Defining Output Control')
        spd = {} 
        for i in range(0,self.nper):
            spd[(i,0)] = ['save head','save budget']
            spd[(i,1)] = ['save head','save budget']
        flopy.modflow.ModflowOc(self.mf,
                                stress_period_data=spd, 
                                compact= True)


    def gmgSolver(self):
        print('Assigning Geometric Multigrid Solver')
        flopy.modflow.ModflowGmg(self.mf, 
                                mxiter=100, 
                                iiter=50, 
                                hclose = 0.32, #too much error as 1
                                rclose= 0.32, 
                                ism=0, 
                                isc=0, 
                                damp=0.3, #had 0.5 before
                                iadamp=1, 
                                ioutgmg=1)                                       
                                #Runs slow try something faster
                                # hclose = 0.5, 
                                # rclose= 0.5, 
                                # ism=1, 
                                # isc=1, 
                                # damp=0.5, 
                                # iadamp=2, 
                                # ioutgmg=1)


    def hob(self):
        print('Importing Well Data with HOB Package')
        # Use this as starting point
        # Get well locations from shapefile
        p = gpd.read_file('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/MW location shapefile/mw_meters.shp')
        # Set up grid intersection with model grid
        ix = flopy.utils.gridintersect.GridIntersect(self.mf.modelgrid,
                                                     method='vertex')
        p['MW#'] = np.arange(1,21)
        
        # Get row,column coordinates of intersections for each well point geometry
        # Note you have to figure out what layers go with it: 7-14m is layers 21-43, midpoint is layer 32
        rc=list() #list of rows/columns per observation
        for i in range(0,len(p.geometry)):
            rc.append(ix.intersects(p.geometry[i])[0][0]) #need list of tuples; add layers of interest to tuple
            
        #Get WL data for dates within model range
        WL_m = pd.read_csv('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/BOW-MW-ALL-DATA-COMPILED_032222.csv')
        #Get rid of nans
        WL_m = WL_m[WL_m['DTW (feet)'].notna()]
        WL_m['Sampling Date'] = pd.to_datetime(WL_m['Sampling Date'])
        WL_m.index=WL_m['Sampling Date']
        
        WL_m['WL_m']=0
        for i in range(1,21):
            TOC = p[p['MW#']==i]['z'].to_numpy()
            new_value = (TOC - WL_m[WL_m['MW#']==i]['DTW (feet)'])/3.281 
            WL_m.loc[WL_m['MW#']==i,'WL_m'] = new_value #convert depth to water to head in meters 
        
        def date_subset(dataframe,start_date,end_date):
            after_start_date = dataframe.index >= start_date
            before_end_date = dataframe.index <= end_date
            between_two_dates = after_start_date & before_end_date
            
            return dataframe.loc[between_two_dates]
        
        start_date = '2016-07-01'
        end_date = '2022-07-01' #'2018-10-31'
        index = pd.date_range(start = start_date, end = end_date, freq='2M') #Try 2 month time steps??? 2M
        WL_m = date_subset(WL_m, start_date, end_date)[['MW#','Sampling Date','WL_m']]
        
        WL_m['sp'] = -999
        WL_m['totim'] = -999
        #Find nearest stress period
        for i in WL_m.index.unique():
            WL_m.loc[WL_m['Sampling Date']==i,'sp'] = np.argmin(abs(i-index))
            WL_m.loc[WL_m['Sampling Date']==i,'totim'] = (i-index[0]).days
        
        # # Save WL_m data for UCODE
        # WL_m['row'] = 999
        # WL_m['col'] = 999
        # for i in range(1,21):
        #     WL_m.loc[WL_m['MW#']==i, 'row'] = rc[i-1][0]
        #     WL_m.loc[WL_m['MW#']==i, 'col'] = rc[i-1][1]
        # WL_m.to_csv('hobsdata.csv')
        
        #Put together list of hobs
        #Note: run Q_weighted_by_kzone.py to get k_data_well
        #Let's try just using middle of well screen for now
        hobs = []
        for i in range(0,len(WL_m)):
            hobs.append(flopy.modflow.mfhob.HeadObservation(self.mf,
                                                            obsname='W'+ str(int(WL_m['MW#'][i])) + '_' + str(WL_m['Sampling Date'][i].strftime('%Y%m%d')),
                                                            layer= 32, 
                                                            row= rc[int(WL_m['MW#'][i]-1)][0],
                                                            column=rc[int(WL_m['MW#'][i]-1)][1],
                                                            irefsp= WL_m['sp'][i], #references stress period
                                                            itt=1, #specified head
                                                            time_series_data= [WL_m['totim'][i], WL_m['WL_m'][i]], #[simulation time, observed head]
                                                            ))
        
        flopy.modflow.mfhob.ModflowHob(self.mf, 
                                       iuhobsv = 40,
                                       obs_data = hobs, 
                                       filenames = [self.mf.name+'.hob',self.mf.name+'.hob_out'])
                     
            

    def runModel(self):
        print('************** Writing Input Files **************')
        self.mf.write_input()
        #print('************** Running the Model **************')
        #mf.run_model(report=True)
        print('************** Pickling the Model **************')
        self.object_save(self.mf,
                    f'{self.workingDir}/mf')

    def testRun(self):
        self.defineModel()
        self.descrit()
        self.basic()
        self.layer_property()
        self.recharge()
        self.wel(self.rech)
        self.load_chd_data()
        self.chd(self.v_grad,self.rch_df)
        self.outputControl()
        self.gmgSolver()
        self.hob()
        
        
        
class mt3d(object):
    """
    Bowman MT3DMS Model
    """
    def __init__(self):
        self.workingDir = '/Users/spencerjordan/Documents/modflow_work'
        self.modelname = 'Bowman_Modflow_Recharge'
        self.model_ws = '/Users/spencerjordan/Documents/modflow_work/recharge_model/'
        ## Start date of the model
        self.startDate = pd.to_datetime('1987-09-01')
        ## End date of the model
        self.endDate = pd.to_datetime('2099-08-31')
        self.agmarDate = pd.to_datetime('2022-05-03')
        self.AgMAR_interval = 10
        
    def object_save(self,obj=object,name=str):      
        file_to_store = open(name+'.pickle', 'wb')
        pickle.dump(obj, file_to_store)
        file_to_store.close()
        
        
    def object_load(self,name=str):
        file_to_read = open(name+'.pickle', "rb")
        obj = pickle.load(file_to_read)
        file_to_read.close()
        return obj
    
    
    def loadMf(self):
        print('Load Existing MODFLOW Model via Pickle? (yes/no)')
        resp = input(':')
        if resp == 'yes':
            ## Can use this to skip the below step 
            print('**** Loading Existing Model ****')
            mf = self.object_load(f'{self.workingDir}/mf')
        elif resp == 'no':
            print('Loading .nam File From Bowman Sim....')
            ## Do this once
            mf = flopy.modflow.Modflow.load('/Users/spencerjordan/Documents/modflow_work/recharge_model/Bowman1404x1092_11_28_rch.nam') # load existing modflow file
            mf.change_model_ws('/Users/spencerjordan/Documents/modflow_work/recharge_model')
            mf.exe_name = '/Users/spencerjordan/Documents/pymake_modflow/examples/mf2005'
            #mf.write_input()
            #Save python object
            self.object_save(mf,f'{self.workingDir}/mf')
            print('**** Pickling Simulation ****')
        ## Need to create the lmt6 file, and then when MODFLOW is ran the mt3d_link.ftl file will be created
        print('Has MODFLOW already been ran in conjuction with link file? (yes/no)')
        modflow = input(':')
        if modflow == 'no':
            print('**** Running MODFLOW Sim for Input to MT3D ****')
            lmt = flopy.modflow.ModflowLmt(mf,
                                       output_file_name='mt3d_link.ftl')
            ## Have to do this no matter what
            lmt.write_file()
            mf.write_name_file()
            mf.run_model()
        else:
            print('**** Skipping MODFLOW simulation and writing of link file ****')
            pass
        modelname_mt = mf.name + "_mt"
        model_ws = mf.model_ws
        exe_name_mt = '/Users/spencerjordan/Documents/pymake_modflow/examples/mt3dms'
            
        mt = flopy.mt3d.Mt3dms(
            modelname=modelname_mt,
            model_ws=model_ws,
            exe_name=exe_name_mt,
            modflowmodel=mf)
        self.mt = mt
        self.mf = mf
        
        
    def basicTransport(self):
        print('**** Initializing Basic Transport Package ****')
        ncomp = 1 # Number of chemical species in simulation
        tunit = 'D' # Days
        lunit = 'M' # Meters
        munit = 'KG' # Note kg/m^3 is the same as mg/cm^3
        #prsity = mf.lpf.sy.array #Effective porosity -> get Sy array from lpf
        porosity_vals = [0.28, 0.30, 0.35, 0.35]
        lpf = self.mf.get_package('LPF')
        #Or recreate tsim from lpf
        tsim = np.stack(lpf.hk.get_value())
        reference = np.sort(np.unique(tsim))[::-1] #sort from largest to smallest
        for i in range(0,4):
            tsim[tsim==reference[i]] = i+1
        prsity = np.zeros(np.shape(tsim))
        for i in range(0,4):
            prsity[tsim==(i+1)] = porosity_vals[i]
        icbund = 1 #solute concentration type (1 is active)
        #Set CHD cells as constant concentration? At least for inflow sides. Not sure what to do about outflow sides...?
        sconc = .004 #starting concentration of solute in kg/m^3 (13 mg/L)
        ifmtcn = 0 #print concentrations in wrap form =1
        nprs = self.mf.nper #frequency of output. >0 = saved at time specified in timprs
        ################################################################################################################################
        ## Updating for Recharge --> Should look into this some more, done quickly on the fly
        nprs_t = 672
        timprs_orig = np.arange(0,nprs_t)*(365.25*56)/nprs_t #total elapsed time --> Spencer Update 11/15
        timprs_1 = timprs_orig[0:208]
        timprs_rch = np.array([x+1 for x in range(59)] + timprs_1[-1])
        timprs_2 = timprs_orig[208::] + 59
        timprs = np.concatenate([timprs_1,timprs_rch,timprs_2])
        #timprs = np.arange(0,nprs)*(365.25*30)/nprs #total elapsed time at which simulation results are saved (number of entries must = nprs)
        #obs = #layer, row, column where concentration is printed at every transport step
        #nprobs = #how frequently concentration at observation points is saved
        ######### May need to lower this ###########
        dt0 = 1 #initial transport step size within each time step of flow soln
        mxstrn = 100000 #max number of transport steps allowed within one time step of flow soln
        ttsmult = 2 #multiplier for successive transport steps within flow time-step
        ttsmax = 30 #max transport step size allowed
        
        flopy.mt3d.Mt3dBtn(
            model = self.mt,
            ncomp = ncomp,
            tunit = tunit,
            lunit = lunit,
            munit = munit,
            prsity = prsity,
            icbund = icbund,
            sconc = sconc,
            ifmtcn = ifmtcn,
            nprs = nprs,
            timprs = timprs,
            #obs = obs,
            #nprobs = nprobs,
            dt0 = dt0,
            mxstrn = mxstrn,
            ttsmult = ttsmult,
            ttsmax = ttsmax)

        
    def advectiveTransport(self):
        print('**** Initializing Advective Transport Package ****')
        mixelm = 0 #0=standard finite-difference method,2=MMOC, 3=HMOC, -1=TVD Ultimate
        percel = 1 #Courant number: number of cells advection allowed in any direction in one transport step
        nadvfd = 0 #which weighting scheme is used: 0 or 1 = upstream weighting, 2 = central-in-space weighting
        itrack = 1  #partickle-tracking algorithm used, 1= first-order Euler 
        wd = 0.5 #concentration weighting factor, 0.5 default. 1= max when advection dominant
        dceps = 1e-5 #small relative cell concentration gradient below which advective transport is considered (number came from an example)
        nplane = 0 #whether random or fixed pattern selected for initial placement of particles, 0= random
        npl = 0 #number of particles per cell placed at cells where concentration gradient <dceps; usually 0?
        nph = 4 ##number of particles per cell placed at cells where concentration gradient >dceps
        npmin = 0 #minimum number of particles allowed per cell
        npmax = 8 #max number of particles per cell
        nlsink = 0 #whether to place random or fixed pattern particles to approximate sink cells in MMOC scheme, usually set to NPLANE
        npsink = 4#number of particles used to approximate sink cells, usually set to NPH
        dchmoc = 1e-3 #critical relative concentration gradient controlling use of either MOC or MMOC in HMOC.
        
        flopy.mt3d.Mt3dAdv(
            model = self.mt,
            mixelm=mixelm,
            percel = percel,
            nadvfd = nadvfd,
            itrack = itrack,
            wd = wd,
            dceps = dceps,
            nplane = nplane,
            npl = npl,
            nph = nph,
            npmin = npmin,
            npmax = npmax,
            nlsink = nlsink,
            npsink = npsink,
            dchmoc = dchmoc)


    def dispersion(self):
        print('**** Initializing Dispersion Package ****')
        ## Dispersion package
        al = 6 # Longitudinal dispersivity for every cell. You can set different dispersivity for every cell if you want.
        trpt = 0.1 # Ratio between horizontal transverse dispersivity to longitudinal dispersivity
        trpv = 0.01 # Ratio of vertical transverse dispersivity to longitudinal dispersivity -> 0.1 or 0.01?
        dmcoef = 8.64E-5 # Nitrate molecular diffusion coefficient m^2/d; just converted 1e-9 m^2/s to m^2/d
        flopy.mt3d.Mt3dDsp(self.mt, 
                           al=al, 
                           trpt=trpt, 
                           trpv=trpv, 
                           dmcoef=dmcoef)
        ## Import recharge concentrations from HYDRUS (note: check units: should be in mg/cm^3 or kg/m^3)
        with open('/Users/spencerjordan/Documents/Hydrus/python_scripts/agmar_pickle_files/nitrate.p','rb') as handle:
            nitrate_conc = pickle.load(handle)
        ## Convert from mg/L to kg/m3 
        nitrate_conc['C'] = nitrate_conc['C'] / 1000
        ## On monthly interval, change to bimonthly
        ## All of the numeric conversions done for better processing times
        nitrate_conc.sp = (nitrate_conc.sp/2).round(0).astype('int')
        nitrate_conc.index = nitrate_conc.index.astype(int)
        nitrate_conc['sp'] = pd.to_numeric(nitrate_conc['sp'])
        nitrate_conc['row'] = pd.to_numeric(nitrate_conc['row'])
        nitrate_conc['col'] = pd.to_numeric(nitrate_conc['col'])
        ## No unit conversion necessary--> mg/cm3 == kg/m3
        nitrate_conc['C'] = pd.to_numeric(nitrate_conc['C'])
        nitrate_conc = nitrate_conc.groupby(by=['sp','row','col'], as_index=False).mean(numeric_only=True)
        nitrate_conc.col = nitrate_conc.col -1
        nitrate_conc.row = nitrate_conc.row -1
        #c_rch_R.sp = c_rch_R.sp -1
        #######################################################################
        ## Edit the SP data to prepare for the inclusion of the AgMAR floodings
        #######################################################################
        ## First stress period to replace with AgMAR data
        ## Hardcoding for now to get the 10-yr simulation running
        start = 208
        for x in range(7+1):
            nitrate_conc.loc[nitrate_conc['sp']>start,'sp'] = nitrate_conc.loc[nitrate_conc['sp']>start,'sp'] + 58
            nitrate_conc = nitrate_conc[nitrate_conc['sp'] != start]
            ## Number of SPs until next flooding
            if x == 0:
                start += 57 + 58
            else:
                start += 59 + 58
        ##################################################
        ## Read in the AgMAR recharge datasets from HYDRUS
        ##################################################
        ## Loop through for each flooding, +1 added becuase numAgmar is number of floodings after May 2022 flood
        start = 208
        for i in range(7+1):
            print(f'**** applying agmar {i} ****')
            with open(f'/Users/spencerjordan/Documents/Hydrus/python_scripts/agmar_pickle_files/nitrate_agmar_{i}.p','rb') as handle:
                nitrate_conc_agmar = pickle.load(handle)
            ## Convert from mg/L to kg/m3 
            nitrate_conc_agmar['C'] = nitrate_conc_agmar['C'] / 1000
            #On monthly interval; fix to bimonthly
            ## All of the numeric conversions done for better processing times
            nitrate_conc_agmar.index = nitrate_conc_agmar.index.astype(int)
            nitrate_conc_agmar['sp'] = pd.to_numeric(nitrate_conc_agmar['sp'])
            nitrate_conc_agmar['row'] = pd.to_numeric(nitrate_conc_agmar['row'])
            nitrate_conc_agmar['col'] = pd.to_numeric(nitrate_conc_agmar['col'])
            ## No unit conversion necessary--> mg/cm3 == kg/m3
            nitrate_conc_agmar['C'] = pd.to_numeric(nitrate_conc_agmar['C'])
            nitrate_conc_agmar = nitrate_conc_agmar.groupby(by=['sp','row','col'], as_index=False).mean(numeric_only=True)
            nitrate_conc_agmar.col = nitrate_conc_agmar.col -1
            nitrate_conc_agmar.row = nitrate_conc_agmar.row -1
            ## Edit to SP's to match that of the original model
            if i == 0:
                nitrate_conc_agmar['sp'] = nitrate_conc_agmar['sp'] + 207
            elif i == 1:
                nitrate_conc_agmar['sp'] = nitrate_conc_agmar['sp'] + 207 + 58 + 57
            else:
                nitrate_conc_agmar['sp'] = nitrate_conc_agmar['sp'] + 207 + (i * 58) + 57 + (59 * (i - 1))
            ## Merge the original and AgMAR datasets together
            ## Merge the original and AgMAR datasets together
            nitrate_1 = nitrate_conc[nitrate_conc.sp<start]
            nitrate_2 = nitrate_conc[nitrate_conc.sp>start]
            nitrate_conc = pd.concat([nitrate_1,nitrate_conc_agmar,nitrate_2])
            ## Number of SPs until next flooding
            if x == 0:
                start += 57 + 58
            else:
                start += 59 + 58
        nitrate_conc = nitrate_conc.reset_index(drop=True)
        self.nitrate_conc = nitrate_conc
            
            
    def itype(self):
        print('**** Initializing crch package ****')
        ## Concentration of solute in recharge. dictionary of arrays
        crch = dict() 
        ## Convert the nitrate_conc DataFrame into Flopy format
        for i in self.nitrate_conc.sp.unique():
            sub = self.nitrate_conc.loc[self.nitrate_conc['sp']==i,:]
            row = np.array(sub.col) 
            col = np.array(sub.row) 
            data = np.array(sub.C)
            grid = np.zeros((117,91))
            grid[row,col] = data
            crch[i] = grid
        ## Assign concentrations based on CHD sources
        ## Need to know layer, row, column of CHD in all stress periods - use mf.chd.stress_period_data rec.array
        itype = flopy.mt3d.Mt3dSsm.itype_dict()
        self.crch = crch
        self.itype = itype

    
    def ssm(self):
        print('**** Initializing ssm package ****')
        ssm_data = {} 
        for i in tqdm(range(0,self.mf.nper)):
            sub = self.mf.chd.stress_period_data[i]
            sub.dtype.names = ('k','i','j', 'css','itype')
            # #Exclude outflow sides: W boundary (column 0), S boundary (row 116)
            # sub = sub[(sub.i!=116) & (sub.j!=0)]
            new_dt = np.dtype(sub.dtype.descr + [('ccms', '<f4')]) #Add new field to rec.array
            ccms = np.full(sub.shape[0], 0, dtype=[('ccms', '<f4')])
            sub = rfn.merge_arrays((sub, ccms), flatten=True)
            spd = list()
            for j in range(0,len(sub)):
                spd.append([sub['k'][j], sub['i'][j], sub['j'][j],
                            0.004, #css (incoming concentration of 13 mg/L - or 4mg/L?)
                            self.itype['CHD']]) 
            ssm_data[i] = spd
        ## Call flopy object
        flopy.mt3d.Mt3dSsm(model = self.mt,
                           crch = self.crch,
                           stress_period_data = ssm_data)
        
        
    def gcg(self):
        print('**** Initializing generalized conjugate gradient package ****')
        mxiter = 1 # max number of outer iterations; should be >1 when nonlinear sorption isotherm is included
        iter1 = 100 # max inner iterations
        isolve = 3 # MIC converges faster but uses more memory
        ncrs = 1 # approximate dispersion tensor cross terms
        cclose = 1e-3 # convergence criterion for relative concentration (1e-5 suggested)
        flopy.mt3d.Mt3dGcg(
            model = self.mt, 
            mxiter=mxiter,
            iter1 = iter1,
            isolve = isolve,
            ncrs = ncrs,
            cclose = cclose)
        
        
    def runMT3D(self):
        print('**** Writing Input Files ****')
        self.mt.write_input()
        print('**** Running MT3D Sim ****')
        self.mt.run_model()
        print('**** Pickling MT3D Output ****')
        self.object_save(self.mt,f'{self.workingDir}/mt')
        
        
    def runModel(self):
        print('**** Welcome to the Bowman MT3D Model ****')
        self.loadMf()
        self.basicTransport()
        self.advectiveTransport()
        self.dispersion()
        self.itype()
        self.ssm()
        self.gcg()
        self.runMT3D()
    
    