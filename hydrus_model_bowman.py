#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:32:38 2023

A collection of classes for the Bowman HYDRUS-1D model
    - Generates ATMOSPH.IN files
    - Runs Single and Monte Carlo Simulations
    - Generates result figures
    - Interpolates results for MODFLOW/MT3DMS inputs

@author: spencerjordan
"""
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from scipy.interpolate import griddata
import time
import math

###############################################################################
####################### Create ATMOSPH.IN inputs ##############################
###############################################################################
class atmosph(object):
    """
    Processes the input data from the manual mass balance, along with climate
    data from CIMIS, to generate and write the HYDRUS atmosph.in files.
    
    The only user input is the ET adjustment factor, which is the parameter 
    used to adjust the ETo predicted by CIMIS to better match that from 
    Bowman's Ranch Systems predicted ETo.
    
    CIMIS is underpredicting ET by 22% based on Ranch Systems
    --> See ET_comparisons.py for more details
    """
    ## Path to mass balance .csv file
    PATH_MB = '~/Documents/bowman_data_analysis/bowmanMassBalance/massBalanceMainData.csv'
    ## Path to the cimis_data
    PATH_CIMIS = '~/Documents/Hydrus/python_scripts/cimis_daily_updated.csv'
    

    def __init__(self,et_adjust=0.92,fullCimis=False,agmar_mult=1.0):
        ## Adjustment factor for CIMIS ET
        self.et_adjust = et_adjust
        self.fullCimis = fullCimis
        ## Main atmosph data set
        self.atmosph_data = None
        ## Years that trees are replanted for each block
        self.treeReplantings = {'NE1':[1976,2002,2022,2047,2072],
                                'NE2':[1970,1996,2022,2047,2072],
                                'NW':[1959,1985,2011,2037,2062,2087],
                                'SW1':[1976,2002,2018,2044,2069,2094],
                                'SW2':[1976,2002,2028,2053,2078],
                                'SE':[1958,1985,2010,2036,2061,2086]}
        self.agmar_mult = agmar_mult
        
        
    def load_CIMIS(self,MID=True):
        """
        Load the CIMIS data from Station 71, Modesto CA
        
        Option to load the full extent of climate data or only the portion that
        corresponds to the irrigation/fertilizer data that we have. If we do this,
        we need to recreate the ET calculated by Hanni.
        
        The MID parameter will force the use of Modesto Irrigation District Precip
        values instead of the CIMIS predictions
        """
        print('Loading CIMIS Data....')
        data = pd.read_csv(self.PATH_CIMIS)
        data = data.set_index(pd.to_datetime(data['Date']))
        data['leap'] = data.index.is_leap_year
        ## Choice to use the full extent of the data or only that we have irrigation/fert data for
        if self.fullCimis:
            data = data[data.index>=pd.to_datetime('09/01/1987',format='%m/%d/%Y')]
            data = data[data.index<pd.to_datetime('09/01/2022',format='%m/%d/%Y')]
        else:
            ## Cutting the CIMIS data to the desired range of 2013 to 2022 GS's
            data = data[data.index>=pd.to_datetime('09/01/2012',format='%m/%d/%Y')]
            data = data[data.index<pd.to_datetime('09/01/2022',format='%m/%d/%Y')]
        ## Drop non-essential columns
        data = data[['Precip (mm)','ETo (mm)','MID Precip (cm)']]
        data['MID Precip (mm)'] = data['MID Precip (cm)'] * 10
        if MID:
            data.loc[data.index<pd.to_datetime('09/01/2022'),'Precip (mm)'] = data.loc[data.index<pd.to_datetime('09/01/2022'),'MID Precip (mm)']
        data['ETo (mm)'] = data['ETo (mm)'] * self.et_adjust
        return data
        
    
    def load_mass_balance(self):
        """
        Load the mass balance dataset, maintained externally in Excel
        """
        print('Loading Mass Balance....')
        ## Path to manual mass balance
        path = self.PATH_MB
        ## Irrigation, precipitation, and ET data from mass balance
        mainDat = pd.read_csv(f'{path}',skiprows=1)
        mainDat = mainDat[mainDat['Date'].notna()]
        ## Set datetime index
        index = pd.to_datetime(mainDat['Date'])
        index.name = 'DateTime'
        mainDat = mainDat.set_index(index)
        ## Cutting data at the end of the 2022 growing season
        mainDat = mainDat[mainDat.index<pd.to_datetime('09/01/2022')]
        ## Set numeric values for fert and irrigation --> Necessary for irrigation and fert editing
        for key in ['Precip', 'Irrigation', 'NE1_I', 'NE2_I', 'NW_I','SW1_I', 
                    'SW2_I', 'SE_I', 'NE1_ET', 'NE2_ET', 'NW_ET', 'SW1_ET','SW2_ET', 
                    'SE_ET', 'NE1_fert', 'NE2_fert', 'NW_fert', 'SW1_fert','SW2_fert', 
                    'SE_fert']:
            mainDat[key] = pd.to_numeric(mainDat[key])
        ## Adjust the pre-hflc irrigation to match the post
        mainDat = self.adjust_mb_irrigation(mainDat,adjust=True)
        return mainDat
    
    
    def adjust_mb_irrigation(self,mainDat,adjust):
        """
        Adjust the mass balance irrigation so that pre-HFLC matches the amounts
        and frequencies of post-HFLC
        """
        #####################################################
        ## Setting the pre-HFLC irrigation to match post-HFLC
        #####################################################
        for block in ['NE1','NE2','SW1','SW2','NW','SE']:
            ## Fixing a strange irrigation event that occured on 6/6/19
            temp = mainDat.loc[mainDat.index==pd.to_datetime('6/6/19'),f'{block}_I']
            val = temp.values[0] / 40
            """
            For whatever reason, it looks like there are 70 days where the irrigation
            was recorded as a factor of 10 too low. Multiplying them by 10 makes
            the 2019 data match the other years signifigantly better. 
            """
            idx = pd.to_datetime('6/6/19')
            for i in range(70):
                if i == 0:
                    mainDat.loc[mainDat.index==idx,f'{block}_I'] = val
                else:
                    mainDat.loc[mainDat.index==idx,f'{block}_I'] = mainDat[f'{block}_I'][idx] * 10
                idx = idx - pd.to_timedelta(1,unit='d')
            ############################################################################
            ## Distribute irrigation over three days, using a 0.3, 0.4, 0.3 distribution
            ## Decided analytically, I (Spencer) think the 0.3/0.4/0.3 looks the best
            ############################################################################
            if adjust:
                preSet = mainDat.loc[mainDat.index<pd.to_datetime('03/20/2019'),[f'{block}_I',
                                                                                 f'{block}_fert']]
                newSet = preSet.copy()
                for idx,val in enumerate(preSet[f'{block}_I']):
                    ## Edit the irrigation if it exists
                    if val > 0:
                        newSet[f'{block}_I'][idx] = val * 0.3
                        newSet[f'{block}_I'][idx+1] = val * 0.4
                        newSet[f'{block}_I'][idx+2] = val * 0.3
                        ## If there is ferilizer applied that day, move it to the third
                        if preSet[f'{block}_fert'][idx] > 0:
                            if newSet[f'{block}_fert'][idx+2] > 0:
                                print('**** Error: Overwriting Existing Fert Data ****')
                            newSet[f'{block}_fert'][idx] = 0
                            newSet[f'{block}_fert'][idx+2] = preSet[f'{block}_fert'][idx]
                ## Put the data back in
                mainDat.loc[mainDat.index<pd.to_datetime('03/20/2019'),[f'{block}_I',f'{block}_fert']] = newSet.values   
        return mainDat
    
    
    def adjust_mb_alignment(self,mbData):
        """
        Ensure that fertilizer application dates align with irrigation, otherwise
        we will run into divide by zero errors later
        
        This is especailly the case for the 2018 year, a large number of the 
        fertilizer inputs do not align with irrigation events
        """
        tempSet = mbData.copy()
        for block in ['NE1','NE2','SW1','SW2','NW','SE']:
            #############
            ## Fixing some fertigation dates to align with the irrigation
            #############
            for idx,val in enumerate(tempSet[f'{block}_I']):
                if (tempSet[f'{block}_fert'][idx] > 0) & (val == 0):
                    ############
                    ## First try and align it with an irrigation event that occurs the day after
                    ############
                    if tempSet[f'{block}_I'][idx+1] > 0:
                        fert_val = tempSet[f'{block}_fert'][idx]
                        prev_fert_val = tempSet[f'{block}_fert'][idx+1]
                        tempSet[f'{block}_fert'][idx+1] = fert_val + prev_fert_val
                        tempSet[f'{block}_fert'][idx] = 0
                        #tempSet.loc[tempSet.reset_index().index==idx+1,f'{block}_fert'] = mbData.loc[mbData.reset_index().index==idx,f'{block}_fert']
                        #tempSet.loc[tempSet.reset_index().index==idx,f'{block}_fert'] = 0
                    ############
                    ## Else, add a new irrigation event with 0.5 cm of water
                    ## Picked as a semi-average value of irrigation across the board
                    ############
                    else:
                        tempSet[f'{block}_I'][idx] = 0.5
                        #tempSet.loc[tempSet.reset_index().index==idx,f'{block}_I'] = 0.5
            ## Put the data back in
        return tempSet
    
    
    def createClimateValues(self,cimis_data):
        """
        Calculates the values for the create_climates function automatically, 
        allows for the quick testing of different sets of climate data
        """
        ## Start date to begin repititions
        start = 2022
        ## End date of first repitition
        end = start + (len(cimis_data.index.year.unique()) - 1)
        ## Number of years in the climate data
        numYears = len(cimis_data.index.year.unique()) - 1
        ## Number of iterations for main loop
        mainIts = int(round(141/(round(len(cimis_data)/365,0)),0))
        ## Year to begin concatenating backwards
        concatBackwards = math.ceil((2099-2022)/numYears)
        ## Start date for when we start backwards concatenating
        start2 = 2022 - (len(cimis_data.index.year.unique()) - 1)
        ## Number of extra years to tack on for forwards propogation
        extraForwards = (2099-2022)%numYears
        ## Number of extra years to tack on for backwards propogation
        extraBackwards = (start2-1958)%numYears
        return end,numYears,mainIts,concatBackwards,start2,extraForwards,extraBackwards
        
        
    def create_climates(self,cimis_data):
        """
        Extrapolate the CIMIS climate data out from 1958 until 2099
        
        The reason this looks so complicated is that to deal with leap years, we need to align the 29th of February in the
        right place chronologically when copy/pasting the data
        """
        print('Creating Climate Data....')
        dateRange = pd.date_range(start='09/01/1958',end='08/31/2099')
        dateData = pd.DataFrame({'leap':dateRange.is_leap_year,
                                 'Date':dateRange})
        cimis_data['leap'] = cimis_data.index.is_leap_year
        ## Filler day for when we need to add in Feb 29th
        leapFiller = cimis_data.loc[cimis_data.index==pd.to_datetime('02/29/2016')]
        ## Start going forwards in the ten years after the time-period we have data for
        start = 2022
        #end = 2032
        ## Auto calculate values to perform data propogations
        end,numYears,mainIts,concatBackwards,start2,extraForwards,extraBackwards = self.createClimateValues(cimis_data)
        ## Start with forwards propogation of data
        forwards = True
        for i in range(mainIts):
            ## Set of climate data that will be repeated
            repeating = cimis_data.copy()
            ## Range of dates for this timeframe
            dateIndex = pd.date_range(start=f'09/01/{start}',end=f'08/31/{end}')
            ## Organize into a DataFrame to compare leap years to repeating
            dateData = pd.DataFrame({'Date':dateIndex,
                                     'leap':dateIndex.is_leap_year})
            ## Have to set date as the index or else the zip function does not work
            dateData = dateData.set_index('Date')
            ## Object to loop through
            if len(repeating.index.year.unique()[1:]) == len(dateData.index.year.unique()[1:]):
                dateObj = zip(repeating.index.year.unique()[1:],dateData.index.year.unique()[1:])
            else:
                ## Need 7 on top, only 4 on the bottom
                ## i.e. there is an extra seven years when copying forward and an extra four
                ## when copying backwards, so we only need those amounts, respectively
                if forwards:
                    yearVal = min(cimis_data.index.year.unique()) + extraForwards
                    repeating = repeating[repeating.index<pd.to_datetime(f'09/01/{yearVal}')]
                else:
                    yearVal = min(cimis_data.index.year.unique()) + extraBackwards
                    repeating = repeating[repeating.index<pd.to_datetime(f'09/01/{yearVal}')]
                dateObj = zip(repeating.index.year.unique()[1:],dateData.index.year.unique()[1:])
            ## Reset both indices
            repeating = repeating.reset_index()
            dateData = dateData.reset_index()
            ## Check on the leap years and fix if missaligned
            for year1,year2 in dateObj:
                ## Grab the boolean leap values
                leap_cimis = repeating.loc[repeating['Date']==pd.to_datetime(f'08/01/{year1}'),'leap'].values[0]
                leap_dates = dateData.loc[dateData['Date']==pd.to_datetime(f'08/01/{year2}'),'leap'].values[0]
                ## CIMIS data is leap-year but dates are not --> delete the leap day
                if leap_cimis and not leap_dates:
                    ## Get index of leap day and then drop the value
                    leap_idx = repeating.loc[repeating['Date']==pd.to_datetime(f'02/29/{year1}')].index[0]
                    repeating = repeating.drop(leap_idx)
                ## Dates are leap-year but CIMIS data is not --> add a leap day in the correct place
                elif leap_dates and not leap_cimis:
                    ## Grab the data before and after the leap day that needs to be there, and then concat it in
                    c1 = repeating[0:repeating.loc[repeating['Date']==pd.to_datetime(f'02/28/{year1}')].index[0]+1]
                    c2 = repeating[repeating.loc[repeating['Date']==pd.to_datetime(f'02/28/{year1}')].index[0]+1:]
                    repeating = pd.concat([c1,leapFiller,c2])
                ## Otherwise we should be okay
                else:
                    pass
            ## On first pass, concat with 2012-2022 data
            if i == 0:
                climateData = pd.concat([cimis_data,repeating])
            ## If forwards, concat the new data to the end of climateData
            elif i < concatBackwards:
                climateData = pd.concat([climateData,repeating])
            ## If backwards, concat the new data to the beginning of climateData
            else:
                climateData = pd.concat([repeating,climateData])
            ## Switch from forwards to backwards propogation of data after seven repititions
            if i == concatBackwards-1:
                forwards = False
                start = start2
                end = 2022
            ## Sets proper start and end dates and will change the values when necessary
            ## in order to fit the new timeframe perfectly
            if forwards:
                if i == concatBackwards-2:
                    start += numYears
                    end += extraForwards
                else:
                    start += numYears
                    end += numYears
            else:
                if i == mainIts-2:
                    start -= numYears
                    end -= (numYears - extraBackwards)
                else:
                    start -= numYears
                    end -= numYears
        ## Set the full date range as the index
        climateData = climateData.set_index(dateRange)
        return climateData


    def cleanMassBalance(self,mass_balance):
        """
        Function to "undo" all of the measured data from tree replantings. This
        is done because the edits to ET, I, and Fert are done later by another 
        function. This is viable, especially for the NW and SE edits because 
        prior to swapping to the PH system measurements in 2019, all the blocks
        in the orchard were given exactly the same inputs. So copying over the 
        data from SW2 is fair game and is the same process as was done in the manual
        mass balance.
        """
        ## NE block was replanted in GS 2022, so need some fake data to copy over --> using GS 2021 data
        mass_balance.loc[mass_balance.index>pd.to_datetime('08/31/2021'),['NE1_fert','NE1_ET','NE1_I']] = mass_balance.loc[(mass_balance.index>pd.to_datetime('08/31/2020'))&(mass_balance.index<pd.to_datetime('09/01/2021')),['NE1_fert','NE1_ET','NE1_I']].values
        mass_balance.loc[mass_balance.index>pd.to_datetime('08/31/2021'),['NE2_fert','NE2_ET','NE2_I']] = mass_balance.loc[(mass_balance.index>pd.to_datetime('08/31/2020'))&(mass_balance.index<pd.to_datetime('09/01/2021')),['NE2_fert','NE2_ET','NE2_I']].values
        ## SW1 block was replanted in 2018, so needs SW2 data to propogate
        ## A note here, is that we will use the original SW1 HFLC data as a proxy for all HFLC blocks when replanted
        hflc_replant = mass_balance.loc[mass_balance.index>pd.to_datetime('08/31/2017'),['SW1_fert','SW1_ET','SW1_I']]
        mass_balance.loc[mass_balance.index>pd.to_datetime('08/31/2017'),['SW1_fert','SW1_ET','SW1_I']] = mass_balance.loc[mass_balance.index>pd.to_datetime('08/31/2017'),['SW2_fert','SW2_ET','SW2_I']].values
        mass_balance.loc[mass_balance.index<pd.to_datetime('09/01/2017'),['NW_fert','NW_ET','NW_I']] = mass_balance.loc[mass_balance.index<pd.to_datetime('09/01/2017'),['SW2_fert','SW2_ET','SW2_I']].values
        mass_balance.loc[mass_balance.index<pd.to_datetime('09/01/2017'),['SE_fert','SE_ET','SE_I']] = mass_balance.loc[mass_balance.index<pd.to_datetime('09/01/2017'),['SW2_fert','SW2_ET','SW2_I']].values
        return mass_balance, hflc_replant
    
    
    def populate_data(self,mass_balance):
        """
        Using the entire period of climate and irrigation data, but only the
        fertilizer application under pre-hflc conditions. 
        """
        print('Creating Fert and Irrigation Data....')
        ## Edit the input data some to account for tree-replantings
        mass_balance,hflc_replant = self.cleanMassBalance(mass_balance)
        #######################################################################
        ############################# PRE-HFLC ################################
        #######################################################################
        ## pre-HFLC range of dates
        dateRange_pre = pd.date_range(start='09/01/1958',end='08/31/2012')
        ## Set all of the mass balance fertilizer applications to match pre-hflc
        mb_pre = mass_balance.copy()
        for block in ['NE1','NE2','SW1','SW2','NW','SE']:
            ## Account for leap year in 2016/2020 when copying data over
            ## moving the leap day from 2016 to 2015
            replace = mb_pre.loc[(mb_pre.index>pd.to_datetime('08/31/2012'))&(mb_pre.index<pd.to_datetime('09/01/2017')),f'{block}_fert']
            leapVal = replace.loc[replace.index==pd.to_datetime('02/29/2016')]
            replace = replace.drop(pd.to_datetime('02/29/2016'))
            c1 = replace[replace.index<=pd.to_datetime('02/28/2015')]
            c2 = replace[replace.index>pd.to_datetime('02/28/2015')]
            replace = pd.concat([c1,leapVal,c2])
            mb_pre.loc[mb_pre.index>pd.to_datetime('08/31/2017'),f'{block}_fert'] = replace.values
        #######################################################################
        ############################# POST-HFLC ###############################
        #######################################################################
        ## post-HFLC range of dates
        dateRange_post = pd.date_range(start='09/01/2022',end='08/31/2099')
        ## Set all fo the mass balance fertilizer applications to match pre-hflc
        mb_post = mass_balance.copy()
        for block in ['NE1','NE2','SW1','SW2','NW','SE']:
            ## Account for leap year in 2016/2020 when copying data over
            ## moving the leap day from 2016 to 2015
            replace = mb_post.loc[mb_post.index>pd.to_datetime('08/31/2017'),f'{block}_fert']
            leapVal = replace.loc[replace.index==pd.to_datetime('02/29/2020')]
            replace = replace.drop(pd.to_datetime('02/29/2020'))
            c1 = replace[replace.index<=pd.to_datetime('02/28/2021')]
            c2 = replace[replace.index>pd.to_datetime('02/28/2021')]
            replace = pd.concat([c1,leapVal,c2])
            mb_post.loc[(mb_post.index>pd.to_datetime('08/31/2012'))&(mb_post.index<pd.to_datetime('09/01/2017')),f'{block}_fert'] = replace.values
        ## Ensure that fertilizer applications align with irrigation inputs
        mb_pre = self.adjust_mb_alignment(mb_pre)
        mb_post = self.adjust_mb_alignment(mb_post)
        mass_balance = self.adjust_mb_alignment(mass_balance)
        ## Run the data processor function to extrapolate inputs
        mainDatPre = self.data_processor(mb_pre,dateRange_pre,6,direction='backwards')
        mainDatPost = self.data_processor(mb_post,dateRange_post,8,direction='forwards')
        ## Ensure that all fertilizer applications have an irrgation event, will print errors if not
        self.fertigation_test(mainDatPre)
        self.fertigation_test(mainDatPost)
        self.fertigation_test(mass_balance)
        ## Concat the pre, current, and post data to create the final dataset
        atmosph_data = pd.concat([mainDatPre,mass_balance,mainDatPost])
        return atmosph_data
    
    
    def fertigation_test(self,mb):
        """
        Tests to ensure that every fetilizer application is co-located with irrigation
        Will print info as to which block and where the mis-match occurs
        """
        for block in ['NE1','NE2','SW1','SW2','NW','SE']:
            for idx,val in enumerate(mb[f'{block}_fert']):
                if (val > 0) & (mb[f'{block}_I'][idx] == 0):
                    print(f'******** {block} {idx} ********')
        
    
    def data_processor(self,mass_balance,dateRange,itterations,direction):
        """
        Process the data while accounting for leap years to maintain maximum accuracy
        
        Same process as was done with the climate data but decided to put it into a 
        function. Should update to have the climate data propogate using this as well
        """
        ## pre-HFLC range of dates
        dateRange = dateRange
        ## Check for leap years
        mass_balance['leap'] = mass_balance.index.is_leap_year
        ## Filler leap-day
        leapFiller =  mass_balance.loc[mass_balance.index==pd.to_datetime('02/29/2016')]
        df = pd.DataFrame()
        ## Start and end years of the pre-HFLC data
        if direction == 'backwards':
            start = 2002
            end = 2012
        else:
            start = 2022
            end = 2032
        ## Copy pre-HFLC data X times
        for i in range(itterations):
            if (start < dateRange.min().year) & (direction == 'backwards'):
                code = '1'
                end = dateRange.min().year
            elif (end > dateRange.max().year) & (direction == 'forwards'):
                code = '2'
                end = dateRange.max().year
            else:
                pass
            repeating = mass_balance.copy()
            ## Range of dates for this timeframe
            dateIndex = pd.date_range(start=f'09/01/{start}',end=f'08/31/{end}')
            ## Organize into a DataFrame to compare leap years to repeating
            dateData = pd.DataFrame({'Date':dateIndex,
                                     'leap':dateIndex.is_leap_year})
            ## Have to set date as the index or else the zip function does not work
            dateData = dateData.set_index('Date')
            ## For the final data pasting, we only need one additional year
            if len(repeating.index.year.unique()[1:]) == len(dateData.index.year.unique()[1:]):
                dateObj = zip(repeating.index.year.unique()[1:],dateData.index.year.unique()[1:])
            ## This will depend on the case-use
            else:
                if code =='1':
                    repeating = repeating[repeating.index>pd.to_datetime('08/31/2018')]
                    dateObj = zip(repeating.index.year.unique()[1:],dateData.index.year.unique()[1:])
                else:
                    repeating = repeating[repeating.index<pd.to_datetime('09/01/2019')]
                    dateObj = zip(repeating.index.year.unique()[1:],dateData.index.year.unique()[1:])
            #dateData
            repeating = repeating.reset_index(drop=True)
            repeating['Date'] = pd.to_datetime(repeating['Date'])
            dateData = dateData.reset_index()
            for year1,year2 in dateObj:
                ## Grab the boolean leap values
                leap_mb = repeating.loc[repeating['Date']==pd.to_datetime(f'08/01/{year1}'),'leap'].values[0]
                leap_dates = dateData.loc[dateData['Date']==pd.to_datetime(f'08/01/{year2}'),'leap'].values[0]
                ## CIMIS data is leap-year but dates are not --> delete the leap day
                if leap_mb and not leap_dates:
                    ## Get index of leap day and then drop the value
                    leap_idx = repeating.loc[repeating['Date']==pd.to_datetime(f'02/29/{year1}')].index[0]
                    repeating = repeating.drop(leap_idx)
                ## Dates are leap-year but CIMIS data is not --> add a leap day
                elif leap_dates and not leap_mb:
                    ## Grab the data before and after the leap day that needs to be there, and then concat it in
                    c1 = repeating[0:repeating.loc[repeating['Date']==pd.to_datetime(f'02/28/{year1}')].index[0]+1]
                    c2 = repeating[repeating.loc[repeating['Date']==pd.to_datetime(f'02/28/{year1}')].index[0]+1:]
                    repeating = pd.concat([c1,leapFiller,c2])
                ## Otherwise we should be okay
                else:
                    pass
            ## Concat to the climateData dataFrame
            if direction == 'backwards':
                df = pd.concat([repeating,df])
                start -= 10
                end -= 10
            else:
                df = pd.concat([df,repeating])
                start += 10
                end += 10

        df = df.set_index(dateRange)
        return df
    
    
    def drop_columns(self,model_mass_balance):
        """
        Drops some unnecessary columns of data
        """
        model_mass_balance = model_mass_balance.drop(['Unnamed: 23','Unnamed: 24'],axis=1)
        for block_key in ['NE1','NE2','NW','SW1','SW2','SE']:
            model_mass_balance = model_mass_balance.drop(f'{block_key}_ET',axis=1)
        return model_mass_balance
    
    
    def concat_data(self,model_mass_balance,climate_data):
        """
        Concatenates the repeating climate data with the repeating fertilizer and irrigation data
        """
        self.atmosph_data = pd.concat([climate_data,model_mass_balance],axis=1)


    def apply_Kc(self):
        """
        Apply the Kc values to the ETo data to calculate ETa
        """
        print('Applying Kc Values....')
        self.atmosph_data['ETa (mm)'] = self.atmosph_data['ETo (mm)']
        ## Kc values based on time of year and tree age
        ## Values are from Doll and Shackel, 2015
        ## '_15' refers to first 15 days of month and '_16' refers to 16th day and onwards
        ####################
        ## Shackle Kc Values
        ####################
        Kc_vals = {'1':0.40,
                   '2':0.41,
                   '3_15':0.55,
                   '3_16':0.67,
                   '4_15':0.75,
                   '4_16':0.84,
                   '5_15':0.89,
                   '5_16':0.98,
                   '6_15':1.02,
                   '6_16':1.07,
                   '7':1.11,
                   '8':1.11,
                   '9_15':1.08,
                   '9_16':1.04,
                   '10_15':0.97,
                   '10_16':0.88,
                   '11':0.69,
                   '12':0.43
                   } 
        """
        #######################################################################################
        ## All ITRC values are taken from Zone 12 using drip and micro-spray irrigation methods
        #######################################################################################
        """
        ###############################
        ## ITRC Kc Values - normal year
        ###############################
        Kc_itrc_norm = {'1':0.77/0.73,
                       '2':0.9/2.12,
                       '3':1.68/4.01,
                       '4':2.75/5.56,
                       '5':5.96/7.32,
                       '6':6.39/7.58,
                       '7':6.7/7.98,
                       '8':5.7/6.76,
                       '9':4.32/5.39,
                       '10':2.82/3.47,
                       '11':0.45/1.05,
                       '12':0.87/0.99}
        ############################
        ## ITRC Kc Values - wet year
        ############################
        Kc_itrc_wet = {'1':0.38/0.39,
                       '2':0.83/0.81,
                       '3':2.35/2.76,
                       '4':3.69/4.12,
                       '5':4.15/4.08,
                       '6':5.66/6.31,
                       '7':6.14/7.49,
                       '8':5.76/7.00,
                       '9':3.83/4.78,
                       '10':2.68/3.48,
                       '11':1.00/1.05,
                       '12':0.92/1.02}
        ############################
        ## ITRC Kc Values - dry year
        ############################
        Kc_itrc_dry = {'1':0.64/0.77,
                        '2':1.31/1.24,
                        '3':2.11/2.78,
                        '4':3.78/5.34,
                        '5':5.8/7.14,
                        '6':6.08/7.23,
                        '7':6.31/7.73,
                        '8':5.27/6.38,
                        '9':4.32/5.23,
                        '10':2.47/3.62,
                        '11':0.89/1.26,
                        '12':0.76/1.36}
        
        for month in Kc_vals.keys():
            """
            Go through the climate data and turn ETo to ETa
            """
            ## Check for 15-day split
            if '_' not in month:
                sub = self.atmosph_data.loc[self.atmosph_data.index.month == int(month),'ETo (mm)'] * Kc_vals[month]
                self.atmosph_data.loc[self.atmosph_data.index.month == int(month),'ETa (mm)'] = sub
            else:
                m = month.split('_')[0]
                d = month.split('_')[-1]
                if d == '15':
                    sub = self.atmosph_data.loc[(self.atmosph_data.index.month == int(m))
                                          & (self.atmosph_data.index.day <= int(d)),'ETo (mm)'] * Kc_vals[month]
                    self.atmosph_data.loc[(self.atmosph_data.index.month == int(m))
                                          & (self.atmosph_data.index.day <= int(d)),'ETa (mm)'] = sub
                else:
                    sub = self.atmosph_data.loc[(self.atmosph_data.index.month == int(m))
                                          & (self.atmosph_data.index.day > int(d)),'ETo (mm)'] * Kc_vals[month]
                    self.atmosph_data.loc[(self.atmosph_data.index.month == int(m))
                                          & (self.atmosph_data.index.day > int(d)),'ETa (mm)'] = sub
        ## Rename ET for each block and convert mm to cm
        block_keys = ['NE1','NE2','NW','SW1','SW2','SE']
        for block in block_keys:
            self.atmosph_data[f'{block}_ET'] = self.atmosph_data['ETa (mm)'] / 10


    def create_P_I(self):
        """
        Creat the {block}_P_I columns in atmosph_data DataFrame
        """
        block_keys = ['NE1','NE2','NW','SW1','SW2','SE']
        for key in block_keys:
            ## Add P and I and convert I from mm to cm
            self.atmosph_data[f'{key}_P_I'] = round(self.atmosph_data[f'{key}_I'] + self.atmosph_data['Precip (mm)']/10,3)
            self.atmosph_data[f'{key}_P_I'] = self.atmosph_data[f'{key}_P_I'].fillna(0)
        
        
    def adjust_fert(self):
        """
        Converts fertilizer applied in kg/ha to mg/cm^3 (which HYDRUS needs)
        Also ensures that there is an input precipitation event for every fertilizer application
        """
        block_keys = ['NE1', 'NE2', 'NW', 'SW1','SW2', 'SE']            
        ## Calculate fertilizer concentrations
        for key in block_keys:
            self.atmosph_data[key] = pd.to_numeric(self.atmosph_data[f'{key}_fert'],errors='coerce')
            self.atmosph_data[f'{key}_fert_conc'] = (self.atmosph_data[f'{key}_fert']) / (self.atmosph_data[f'{key}_P_I']*100)
            self.atmosph_data[f'{key}_fert_conc'] = self.atmosph_data[f'{key}_fert_conc'].fillna(value=0)
        
        
    def et_split(self):
        """
        Calculate ET split
        E is 24% of total ET the first day of a P or I event
        Then, decreases linearly to 0% on the seventh day
        """
        print('Splitting ET....')
        ## Need to add in year post HFLC there is zero T and all E
        block_keys = ['NE1','NE2','NW','SW1','SW2','SE']
        for key in block_keys:
            evap = []
            trans = []
            for i,val in enumerate(self.atmosph_data[f'{key}_ET']):
                for k in range(7):
                    ## If precip or I in past eight days, use linear func to calculate ET split.
                    ## Once it breaks, will have correct k index for the split saved
                    if self.atmosph_data[f'{key}_P_I'][i-k] > 0:
                        break
                    ## If no precip or I, put all ET to transpiration
                    else:
                        pass
                e = val * (0.24 - ((0.24/6) * k))
                t = val - e
                evap.append(e)
                trans.append(t)
            self.atmosph_data[f'{key}_E'] = evap
            self.atmosph_data[f'{key}_T'] = trans
        
    
    def ET_adjust(self):
        """
        Adjust total ET amounts for years following tree replantings 
        """
        print('Adjusting ET....')
        def f(age):
            return (-0.0501*age**2) + (0.4882*age) - 0.1526
        ages = [0.5,1,2,3,4,5,6]
        mid_season_Kc = 1.11
        ET_mult = [f(x) for x in ages]
        ET_mult = [x/mid_season_Kc for x in ET_mult]
        ET_mult[0] = 1
        #ET_mult[5] = 1
        for treeKey in self.treeReplantings:
            for year in self.treeReplantings[treeKey]:
                for k in range(6):
                    if k == 0:
                        startDay = pd.to_datetime(f'{year-1}-12-01',format='%Y-%m-%d')
                        endDay = pd.to_datetime(f'{year}-09-01',format='%Y-%m-%d')
                    else:
                        startDay = pd.to_datetime(f'{year+k-1}-09-01',format='%Y-%m-%d')
                        endDay = pd.to_datetime(f'{year+k}-09-01',format='%Y-%m-%d')
                    ## grab the first subset of ET
                    sub = self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{treeKey}_ET']
                    sub =  sub * ET_mult[k]
                    ## Then need to put sub back into the original dataset
                    self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{treeKey}_ET'] = sub
    
        
    def IF_adjust(self):
        """
        Adjust the irrigation and fertilizer amounts for the years following tree
        replantings
        """
        print('Adjusting Irrigation and Fert....')
        irr_mult = [0,0.5,0.5,0.75,1,1,1]
        fert_mult = [0,0.36,0.36,0.45,0.75,0.75,1]
        for block in self.treeReplantings:
            ## These are replanting dates, so trees are removed in December of
            ## the year previous
            for year in self.treeReplantings[block]:
                ## !!!for post HFLC, use SW1 data as a proxy instead of equations
                #if year < 2018:
                for k in range(7):
                    ## Trees are replanted in December, so only want to apply 
                    ## multipliers after that date if k==0
                    if k == 0:
                        startDay = pd.to_datetime(f'{year+k-1}-12-01',format='%Y-%m-%d')
                        endDay = pd.to_datetime(f'{year+k}-09-01',format='%Y-%m-%d')
                    else:
                        startDay = pd.to_datetime(f'{year+k-1}-09-01',format='%Y-%m-%d')
                        endDay = pd.to_datetime(f'{year+k}-09-01',format='%Y-%m-%d')
                    sub_irr = self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_I'] * irr_mult[k]
                    sub_fert =  self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_fert'] * fert_mult[k]
                    self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_I'] = sub_irr
                    self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_fert'] = sub_fert
        
        
    def ET_replantings(self):
        """
        Set T to zero for year following tree replantings
        Differs from 'ET_adjust' as it does not appy any growth factor, this simply
        puts all of the ET into E for the years when no trees are present
        """
        print('Calculating Tree Replantings....')
        for block in self.treeReplantings:
            for year in self.treeReplantings[block]:
                ## Trees are replanted in December, so only want to apply 
                ## multipliers after that date
                startDay = pd.to_datetime(f'{year-1}-12-01',format='%Y-%m-%d')
                endDay = pd.to_datetime(f'{year}-09-01',format='%Y-%m-%d')
                #trans = self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_T']
                #evap = self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_E']
                self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_T'] = 0
                ## Assign 100% of ET into evaporation, since no trees are there to transpire
                ## Hanni did not add E&T together, she simply set Transpiration to zero and kept Evap as only
                ## that small percentage of water following replantings
                #self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<endDay),f'{block}_E'] = trans + evap
        
    
    def final_cleaning(self):
        """
        Final data cleaning to drop inf and na values, as well as adding tAtm column
        """
        ## Fill inf and NaN values
        self.atmosph_data.replace([np.inf, -np.inf], 0, inplace=True)
        ## Drop rows with NaN
        self.atmosph_data.fillna(0,inplace=True)
        ## Create the tAtm column
        self.atmosph_data['tAtm'] = np.linspace(1,51500,51500)
    
    
    def apply_AgMAR(self,mult=1.0):
        """
        Apply the AgMAR flooding to the simulations
        Currently only have data from MW-8 implemented
        """
        print('Applying AgMAR Data....')
        ## For now, only applying the singular AgMAR event from summer of 2022
        ## Daily water applied in cm/day for well 8
        appliedWater = [28.90730341,45.46232524,39.8461859,33.20893031,
                        33.20893031,33.20893031,33.20893031,30.88717831,
                        28.56542632,28.56542632,28.56542632,28.56542632,
                        28.56542632,28.56542632,28.56542632,28.56542632,
                        28.56542632,28.56542632,28.56542632,28.56542632,
                        28.56542632,28.56542632,28.56542632,29.17451579,
                        29.8943488,29.8943488,29.8943488,29.8943488,
                        9.964782934]
        appliedWater = [x*mult for x in appliedWater]
        ## Dynamic in case we want to simulatate flooding over multiple blocks
        agmar_blocks = ['NE2']
        ## Flooding ocurred for a period of 28 days
        ## Dynamic to allow for multiple floodings
        startDays = [pd.to_datetime('05/03/2022',format='%m/%d/%Y')]
        ## Repeat flooding every 10th Winter on January 1st
        for i in range(7):
            startDays.append(pd.to_datetime(f'01/01/20{32+i*10}',format='%m/%d/%Y'))
            
        for block in agmar_blocks:
            for startDay in startDays:
                self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<=(startDay+pd.to_timedelta(28,unit='d'))),f'{block}_P_I'] = self.atmosph_data.loc[(self.atmosph_data.index>=startDay)&(self.atmosph_data.index<=(startDay+pd.to_timedelta(28,unit='d'))),f'{block}_P_I'] + appliedWater
        
    
    def generate_header_footer(self):
        """
        Generate a header and a footer for the atmosph.in files
        """
        filename = '/Users/spencerjordan/Documents/Hydrus/Profiles/future_NIT_core_1/ATMOSPH_reference.txt'
        with open(filename,'r',encoding='latin-1') as f:
            text = f.readlines()
        header = text[:9]
        footer = text[-1]
        header[3] = '\t51500\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n'
        return header,footer
                        
    
    def main(self,agmar=False):
        """
        Runs all the processing functions to create atmosph the input
        Agmar parameter controls whether or not to write AgMAR data to the NE2 
        wells
        """
        cimis_data = self.load_CIMIS()
        mass_balance = self.load_mass_balance()
        climate_data = self.create_climates(cimis_data)
        model_mass_balance = self.populate_data(mass_balance)
        #model_mass_balance = self.apply_mass_balance(mass_balance)
        model_mass_balance = self.drop_columns(model_mass_balance)
        self.concat_data(model_mass_balance,climate_data)
        self.apply_Kc()
        self.ET_adjust()
        self.IF_adjust()
        self.create_P_I()
        self.adjust_fert()
        self.et_split()
        self.ET_replantings()
        ## Quickly, recalculate total ET for each block since they have been edited
        for block in self.treeReplantings:
            self.atmosph_data[f'{block}_ET'] = self.atmosph_data[f'{block}_E'] + self.atmosph_data[f'{block}_T']
        ## If running AgMAR sims, do this
        ## Remember, most of the agmar block is not part of the recharge basin,
        ## so we usually need an AgMAR run AND a normal run in order to get the 
        ## correct recharge for the blocks where the flooding occurs
        if agmar:
            self.apply_AgMAR(mult=self.agmar_mult)
        self.final_cleaning()
        print('******* Successfully created atmosph data *******')
        
        
    def irrigation_eff(self):
        """
        Looking into irrigation efficiencies, can we match past climate data 
        to hypothetical fertigation?
        """
        for block in self.treeReplantings.keys():
            block_eff = self.atmosph_data[f'{block}_I'] - self.atmosph_data[f'{block}_ET']
            self.atmosph_data[f'{block}_eff'] = block_eff
        
        
    def write_inputs(self,agmar=False):
        """
        Write the individual ATMOSPH.IN files
        """
        header,footer = self.generate_header_footer()
        ## If AgMAR, only need to write files for NE2 --> for now at least...
        if agmar:
            ## Run only recharge basin blocks
            block_dict = {'NE2':[6,7,8]}
        else:
            block_dict = {'NE1':[1,2,3],
                          'NE2':[6,7,8],
                          'NW':[11,12,16,17],
                          'SW1':[13,14,18,19],
                          'SW2':[15,20],
                          'SE':[4,5,9,10]}
        ## Constant values for ATMOSPH.IN files
        hCritA = 10000  ## Controls evaporation
        rB = 0
        hB = 0
        ht = 0
        tTop = 0
        tBot = 0
        Ampl = 0
        cBot = int(0)
        for block in block_dict:
            for well in block_dict[block]:
                print(f'Writing input for well {well}')
                with open(f'/Users/spencerjordan/Documents/Hydrus/Profiles/future_NIT_core_{well}/ATMOSPH.IN','w') as f:
                    for line in header:
                        f.write(line)
                    for idx in range(len(self.atmosph_data)):
                        tAtm = self.atmosph_data['tAtm'].iloc[idx]
                        Prec = self.atmosph_data[f'{block}_P_I'].iloc[idx]
                        rSoil = round(self.atmosph_data[f'{block}_E'].iloc[idx],7)
                        rRoot = round(self.atmosph_data[f'{block}_T'].iloc[idx],5)
                        cTop = round(self.atmosph_data[f'{block}_fert_conc'].iloc[idx],8)
                        if cTop == 0:
                            cTop = int(0)
                        f.write(f'\t{tAtm}\t{Prec}\t{rSoil}\t{rRoot}\t{hCritA}\t{rB}\t{hB}\t{ht}\t{tTop}\t{tBot}\t{Ampl}\t{cTop}\t{cBot}\n')
                    for line in footer:
                        f.write(line)
        print('******* File creation successful *******')


###############################################################################
######################## Edit Hydrus input files ##############################
###############################################################################
class editInputs(object):
    """
    Helper functions for editing HYDRUS input files
    """
    
    def __init__(self,nwells=[x for x in range(1,21)]):
        self.nwells = nwells
    
    
    def edit_obsNodes(self,nodes=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,
                 150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,299]):
        """
        Edit the location and number of observation nodes in the model
        """
        ## create the node line
        nodeLine = '\t'
        for node in nodes:
            nodeLine = nodeLine + f'{node}\t'
        nodeLine = nodeLine[:-1]
        for well in self.nwells:
            filename = f'/Users/spencerjordan/Documents/Hydrus/Profiles/future_NIT_core_{well}/PROFILE.DAT'
            with open(filename,'r') as f:
                lines = f.readlines()
                lines[-2] = f'\t{len(nodes)}\n'
                lines[-1] = nodeLine
            with open(filename,'w') as f:
                for line in lines:
                    f.write(line)
                    
                    
    def edit_cRoot(self,cRoot=0.04):
        for well in self.nwells:    
            ## Write the new cRoot_max value to SELECTOR.IN
            selectorFile = f'/Users/spencerjordan/Documents/Hydrus/Profiles/future_NIT_core_{well}/SELECTOR.IN'
            with open(selectorFile,'r') as f:
                lines = f.readlines()
                ## New line with cRoot_max value
                newLine = f'        0                                {cRoot}           0.5\n'
                lines[131] = newLine
            with open(selectorFile,'w') as f:
                f.writelines(lines)
                    
                
###############################################################################
##################### Set params and run the model ############################
###############################################################################
class exec_hydrus(object):
    """
    Class to run HYDRUS with randomly generated MC parameters for alpha, n,
    theta_s, theta_r, and Ks. I have a lot of OBS points to create contours
    after processing the simulation data. Also can be used to calibrate the
    cRoot parameter
    
    Could use some improvement in how this is execuated
    """
    
    def __init__(self,nwells=[x for x in range(1,21)],nsims=20,
                 node_list=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,
                              150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,299],
                 cRoot=False):
        self.nwells = nwells
        self.nsims = nsims
        self.node_list = node_list
        self.working_dir = '/Users/spencerjordan/Documents/Hydrus'
        self.params = None
        self.path = None
        self.cRoot = cRoot


    def load_hydraulic_params(self):
        """
        Use numpy random and the statistical distributions givin in Rosetta to 
        predict a random paramter set
    
        Soil layer %sand/%silt/%clay:
        The last four entries are copies of previous layers, these are 
        necessary to set HYDRUS parameters for these four layers when they
        appear in the root zone of the model. There are decay rate parameters
        that only apply to the soils present in the root zone. 
        """
        from rosetta import rosetta, SoilData
        soils = [[10,75,15],
                 [7.5,47,45.5],
                 [7.5,47,45.5],
                 [10,35,55],
                 [52.5,42.5,5],
                 [52.5,42.5,5],
                 [52.5,42.5,5],
                 [33.3,33.3,33.3],
                 [40,20,40],
                 [20,15,65],
                 [5,5,90],
                 [5,5,90],
                 [65,10,25],
                 [65,10,25],
                 [90,5,5],
                 [90,5,5],
                 [90,5,5],
                 [90,5,5],
                 [90,5,5],
                 [90,5,5],
                 [7.5,47,45.5],
                 [52.5,42.5,5],
                 [33.3,33.3,33.3],
                 [65,10,25],
                 ]
        ## Calculate mean and stdev of VG parameters based on textures
        mean,stdev,codes = rosetta(3,SoilData.from_array(soils))
        ## Initialize the DataFrame
        params = pd.DataFrame(columns=['thr', 'ths', 'Alfa', 'n', 'Ks', 'l'])
        ## For each soil, randomly sample based on statistical distribution
        ## to create a random set of 24 parameter sets (one for each layer)
        for idx,soil in enumerate(mean):
            thr = np.random.normal(soil[0],stdev[idx][0])
            ths =  np.random.normal(soil[1],stdev[idx][1])
            ## Need to anti-log these
            alfa = round(10**np.random.normal(soil[2],stdev[idx][2]),4)
            n = round(10**np.random.normal(soil[3],stdev[idx][3]),3)
            Ks = round(10**np.random.normal(soil[4],stdev[idx][4]),5)
            ## l shape parameter always 0.5
            l = 0.5
            ## append to DataFrame
            params.loc[len(params)] = [thr,ths,alfa,n,Ks,l]
        self.params = params
    
    
    def write_hydraulic_params(self):
        """
        Writes the randomly generated hydraulic parameters to the SELECTOR.IN files
        """
        selectorFile = f'{self.path}/SELECTOR.IN'
        with open(selectorFile,'r') as f:
            selector = f.readlines()
        ## Parameter Block
        paramBlock = selector[26:50]
        ## Layers to be optimized
        ## Formatting vector for parameter input
        value_format_vec = "{0:7.4f}{1:8.4f}{2:8.4f}{3:8.3f}{4:11.5f}{5:8.2f}\n"
        ## Loop through each soil property and assign new values
        for i in range(24):
            p = self.params.loc[i,:]
            ## Split the value string and convert everything to floating point
            vals = [float(x) for x in paramBlock[i].split()]
            vals[0] = p[0]
            vals[1] = p[1]
            vals[2] = p[2]
            vals[3] = p[3]
            vals[4] = p[4]
            vals = value_format_vec.format(*vals)
            paramBlock[i] = vals
        selector[26:50] = paramBlock
        with open(selectorFile,'w') as f:
            for line in selector:
                f.write(line)  
   
    
    def call_hydrus(self):
        """
        Function to call the HYDRUS executable
        """
        ## Path to Hydrus executable
        hydrus_exe_path = '/Users/spencerjordan/Documents/hydrus_make_files/source'
        #hydrus_exe_path = '/Users/spencerjordan/source_code/source'
        ## File to point the executable to correct input/output file directory
        level_01 = '/LEVEL_01.DIR'
        ## Wrtie the file
        with open(hydrus_exe_path+level_01,'w') as f:
            f.write(self.path+'/')
        ## Change path and execute Hydrus
        os.chdir(hydrus_exe_path)
        ## Remove the old error file if it exists
        try:
            os.system('rm Error.msg')
        except:
            pass
        os.system('./hydrus LEVEL_01.DIR')
        os.chdir(self.working_dir)
    
    
    def read_obs_file(self):
        """
        Reads the data from the OBS_NODE.OUT file. The file has fairly strange
        formatting, so a good bit of this function is dedicated to organizing the 
        text into a readable format.
        """
        obs_nodes = self.node_list
        out_file = '/OBS_NODE.OUT'
        ## Very complicated way to determine the headers for the dataframe
        with open(self.path+out_file,'r') as f:
            obs_node_out = f.readlines()
        node_ind = [idx for idx, s in enumerate(obs_node_out) if 'Node' in s][0]
        output_names = obs_node_out[node_ind + 2].split(' ')
        unique_vals = []
        ## Could maybe use 'set', but wanted to avoid alphabetizing
        for val in output_names:
            if val not in unique_vals:
                unique_vals.append(val)
        output_names = unique_vals
        output_names.remove('')
        output_names.remove('\n')
        output_names = output_names[1:]
        output_names_rep = output_names * len(obs_nodes)
        ## Sets a placeholder for each variable for each observation node
        obs_nodes_rep = []
        for node in obs_nodes:
            nodes = [node]*len(output_names)
            obs_nodes_rep = obs_nodes_rep + nodes
        ## Creates the final column names
        output_names_all = ''.join([str(a+'_') + str(b)+' ' for a,b in zip(output_names_rep,obs_nodes_rep)]).split(' ')
        output_names_all.remove('')
        obs_node_out = obs_node_out[(node_ind + 3):]
        ## Generate numpy array from .out file
        all_dat = np.genfromtxt(self.path+'/OBS_NODE.OUT',skip_header=10,skip_footer=1)
        ## Generate the dataframe to be saved
        obs_node_out = pd.DataFrame(data = all_dat,columns=(['Time']+output_names_all))
        ## Set the time column to start at zero
        obs_node_out.Time = pd.to_numeric(obs_node_out.Time)
        obs_node_out.Time = obs_node_out.Time - 1
        ## Drop column if all values are NA
        obs_node_out.dropna(axis=1, how='all')     
        obs_node_out = obs_node_out.reset_index()
        obs_node_out = obs_node_out.drop('index',axis=1)
        return obs_node_out
    
    
    def read_solute(self):
        """
        Read the output from solute1.out file and organize into a DataFrame
        """
        headers = ['Time','cvTop','cvBot','Sum(cvTop)','Sum(cvBot)','cvCh0','cvCh1',
                   'cTop','cRoot','cBot','cvRoot','Sum(cvRoot)','Sum(cvNEql)']
        solute = pd.read_csv(f'{self.path}/solute1.out',
                             skiprows=[0,1,2,3],
                             names=headers,
                             usecols=[x for x in range(13)],
                             delim_whitespace=True,
                             low_memory=False)
        return solute
    
    
    def read_tlevel(self):
        """
        Read the output from T_LEVEL.out file and organize into a DataFrame
        """
        tlevel = pd.read_csv(f'{self.path}/T_LEVEL.OUT',
                             skiprows=[0,1,2,3,4,6,7],
                             delim_whitespace=True,
                             low_memory=False)
        return tlevel
    
    
    def read_alevel(self):
        """
        Read the output from A_LEVEL.out file and organize into a DataFrame
        """
        alevel = pd.read_csv(f'{self.path}/A_LEVEL.OUT',
                             skiprows=[0,1,3,4],
                             delim_whitespace=True,
                             low_memory=False)
        return alevel
    
    
    def run_model(self):
        """
        Call Hydrus and read the obsfile
        Runs the model based on self.nwells and self.nsims only
        Always defaults to selecting a random set of parameters from the MC space
        """
        ## pre-define the output data dictionaries
        obs_node_out = {}
        alevel_out = {}
        tlevel_out = {}
        solute_out = {}
        ## Path to the main project folder
        projectPath = '/users/spencerjordan/Documents/Hydrus/Profiles'
        ## Loop through each well and each MC simulation
        for well in self.nwells:
            self.path = f'{projectPath}/future_NIT_core_{well}'
            for sim in range(self.nsims):
                print(f'Running well {well}, MC {sim+1}')
                ## Load random hydraulic params and run hydrus
                self.load_hydraulic_params()
                self.write_hydraulic_params()
                self.call_hydrus()
                ## Ran into file read/write errors --> Needs a quick nap :)
                time.sleep(2)
                ## Load all of the output data
                try:
                    alevel = self.read_alevel()
                    tlevel = self.read_tlevel()
                    solute_data = self.read_solute()
                    obs_data = self.read_obs_file()
                    ## Save the output data in respective dictionaries
                    solute_out[sim] = solute_data
                    obs_node_out[sim] = obs_data
                    tlevel_out[sim] = tlevel
                    alevel_out[sim] = alevel
                except:
                    print('***** RESULT FILES NOT LOADED CORRECTLY *****')
            ## Save the dictionaries holding info for each mc run as pickle files
            ## Save to the cRoot directory if running cRoot calibration
            if not self.cRoot:
                pickle.dump(obs_node_out, open(f'{projectPath}/result_files/flux{well}.p', "wb" ))
                pickle.dump(solute_out, open(f'{projectPath}/result_files/solute{well}.p', "wb" ))
                pickle.dump(tlevel_out, open(f'{projectPath}/result_files/tlevel{well}.p', "wb" ))
                pickle.dump(alevel_out, open(f'{projectPath}/result_files/alevel{well}.p', "wb" ))
            ## Only need solute data for cRoot calibration
            else:
                pickle.dump(solute_out, open(f'{projectPath}/result_files/cRoot/solute{well}.p', "wb" ))
                

    def calibrate_cRoot(self):
        """
        Function to calibrate the cRoot_max parameter by minimizing the RSS 
        between measured and modeled NUE_growth. Where NUE_growth is the N used 
        for tree growth / applied N
        
        Going to use the exec hydrus class to accomplish this
        """
        import os
        ## Dict to hold final RSS error values
        RSS_dict = {}
        ## Load the mass balance to calculate measured NUE_growth
        path = '/Users/spencerjordan/Documents/bowman_data_analysis/N_mass_balance/manual_mass_balance_2022.csv'
        N_balance = pd.read_csv(path)
        N_balance['NUE_growth'] = (N_balance['Uptake kg/ha'] + N_balance['Growth']) / N_balance['Fert kg/ha']
        NUE_measured = N_balance[['block','GS','NUE_growth']]
        ## NE is NaN in GS 2022 because of replanting
        NUE_measured = NUE_measured.fillna(0)
        ## Possible values of cRoot_max --> in intervals of 0.005
        cRoot_vals = [str(round((x/1000),3)) for x in np.linspace(45,80,4)]
        ## Loop through each value, write the value, run hydrus, then compare the outputs
        for cRoot in cRoot_vals:
            ## Go through each well
            for well in self.nwells:    
                ## Write the new cRoot_max value to SELECTOR.IN
                selectorFile = f'{self.working_dir}/Profiles/future_NIT_core_{well}/SELECTOR.IN'
                with open(selectorFile,'r') as f:
                    lines = f.readlines()
                    ## New line with cRoot_max value
                    newLine = f'        0                                {cRoot}           0.5\n'
                    lines[131] = newLine
                with open(selectorFile,'w') as f:
                    f.writelines(lines)
            ## Now run a model with 3 MC runs with the new cRoot_max value --> should be more, but computationally expensive
            self.runHydrus(3)
            hydrus = True
            ## Load the results
            ph = plot_hydrus(path='/Users/spencerjordan/Documents/Hydrus/Profiles/result_files/cRoot')
            ## Janky way to wait until terminal is done running Hydrus
            while hydrus:
                try:
                    ## Load the solute results
                    solute_data = ph.clean_data(ph.load_data(var='solute'))
                    hydrus = False
                    print('HYDRUS Simulations Completed')
                    os.system('rm /Users/spencerjordan/Documents/Hydrus/Profiles/result_files/cRoot/*.p')
                except:
                    pass
            ## Load the applied fertilizer data
            cvTop = ph.process_data(quantiles=[0.5],dataset=solute_data,loadVar='Sum(cvTop)').diff().resample('12MS').sum()
            ## Load the root uptake data
            cvRoot = ph.process_data(quantiles=[0.5],dataset=solute_data,loadVar='Sum(cvRoot)').diff().resample('12MS').sum()
            start = pd.to_datetime('08/31/2012')
            end = pd.to_datetime('09/01/2022')
            ## Organize data per block and slice to desired timeframe
            cvTop = ph.organizeBlocks(cvTop)
            fert = cvTop.set_index(pd.to_datetime(cvTop.index))[(cvTop.index>start)&(cvTop.index<end)]
            cvRoot = ph.organizeBlocks(cvRoot)
            treeUptake = cvRoot.set_index(pd.to_datetime(cvRoot.index))[(cvRoot.index>start)&(cvRoot.index<end)]
            NUE_modeled = pd.DataFrame(treeUptake.values / fert.values,
                                        columns=fert.columns,
                                        index=fert.index)
            ## Holds the RSS for each block for each run of cRoot
            RSS_blocks = {}
            ## Holds the modeled and measured values to make a plot
            ## Want pre and post HFLC seperated for plotting
            Modeled_pre = []
            Measured_pre = []
            Modeled_post = []
            Measured_post = []
            ## Compare measured and modeled  using residual sum of squares
            for block in ['NE','SW','SE','NW']:
                measured = NUE_measured['NUE_growth'][NUE_measured['block']==block]
                modeled = NUE_modeled[block]
                ## Drop data with immature trees
                if block == 'NE':
                    measured = measured[:-1]
                    modeled = modeled[:-1]
                ## Storing the RSS values
                RSS = sum((measured.values - modeled.values)**2)
                RSS_blocks[block] = RSS
                ## Save the modeled and measured data for plotting
                ## Pre
                for x in list(modeled.values[:5]):
                    Modeled_pre.append(x)
                ## Post
                for x in list(modeled.values[5:]):
                    Modeled_post.append(x)
                ## Pre
                for x in list(measured.values[:5]):
                    Measured_pre.append(x)
                ## Post
                for x in list(measured.values[5:]):
                    Measured_post.append(x)
            ## Get mean RSS value across blocks
            RSS_mean = round(np.mean([x for x in RSS_blocks.values()]),3)
            ## Plot measured against modeled data
            fig,ax = plt.subplots()
            ax.set_title(f'Model v. Observed Efficiency for growth\ncRoot = {cRoot}, RSS = {RSS_mean}',
                         fontsize=11)
            ax.set_ylabel('Modeled')
            ax.set_xlabel('Observed')
            ax.axline((0, 0), slope=1,color='black',ls='dashed')
            ## Plot pre and post data
            ax.scatter(Measured_pre,Modeled_pre,color='red',
                       s=6,
                       label='Pre-HFLC')
            ax.scatter(Measured_post,Modeled_post,color='blue',
                       s=6,
                       label='HFLC')
            ax.legend()
            plt.show()
            ## Store the RSS for each block in the RSS_dict
            RSS_dict[cRoot] = RSS_blocks
        return RSS_dict
            
    
    def runHydrus(self,numSims):
        """
        Function to run multiple instances of HYDRUS in Parallel for calibrating
        cRoot
        """
        groups = {
                  1:'1,2,3,4,17',
                  2:'5,6,7,8,18',
                  3:'9,10,11,12,19',
                  4:'13,14,15,16,20'
                 }
        for i in groups:
            wells = groups[i]
            os.system(f"osascript -e 'tell app \"Terminal\" to do script \"cd ~/Documents/Hydrus/python_scripts && conda activate hydrus-modflow && python exec_hydrus_cRoot.py <<< {wells} <<< {numSims} \"'")
    
    
###############################################################################
########################## Plot Hydrus outputs ################################
###############################################################################
class plot_hydrus(object):
    """
    Reads the Pickled HYDRUS outputs created by the exec_hydrus class in order to 
    creates output plots.
    """
    def __init__(self, path='/Users/spencerjordan/Documents/Hydrus/Profiles/result_files/MID_0_92_cRoot_0_040',
                 nwells=[x for x in range(1,21)],
                 result_length=51499):
        """
        Use '/Users/spencerjordan/Documents/Hydrus/Profiles/result_files/et_atmosph_updated' 
        for Spencer's updated results
        """
        ## Path to the pickled result files --> don't love the way I'm doing this right now
        self.path = path
        ## Wells of interest
        self.nwells = nwells
        ## Number of timesteps in the hydrus simulations
        self.result_length = result_length
        ## Titles for the plots, relating each title to a specific plotting variable 
        self.title_dict = {'Conc_299':'Modeled Leaching Nitrate Concentration',
                      'Flux_299':'Modeled Groundwater Recharge',
                      'Sum(cvRoot)':'Cumulative Root Solute Uptake',
                      'cvRoot':'Root Solute Uptake',
                      'theta_299':'Water Content'}
        ## Same as above but for ylabels
        self.ylabel_dict = {'Conc_299':'Concentration [mg/L]',
                      'Flux_299':'Estimated Recharge [cm]',
                      'Sum(cvRoot)':'Solute Uptake',
                      'cvRoot':'Solute Uptake',
                      'theta_299':'Water Content'}
        ## Turn grid on as the default for plt plots
        plt.rcParams['axes.grid'] = True
    
    
    def load_data(self,var):
        """
        Read the pickled data from the OBS_NODE.OUT for solute1.out files
        'var' specifies either flux or solute dataset to be loaded
        """
        data = {}
        ## Load nodeinfo data
        for i in self.nwells:
            with open(f'{self.path}/{var}{i}.p','rb') as handle:
                data[i]= pickle.load(handle)
        return data
    
    
    def clean_data(self,dataset):
        """
        Removes data from simulations that did not converge
        """
        print('Cleaning Data....')
        for j in dataset.keys():
            del_list = []
            ## If the length of the data is shorter than it should be, delete it
            for k in dataset[j].keys():
                if len(dataset[j][k]['Time']) < self.result_length:
                    del_list.append(k)
            for d in del_list:
                del dataset[j][d]
        return dataset
            
            
    ###########################################################################
    ## Process the HYDRUS output
    ###########################################################################
    def process_data(self,
                     dataset,
                     loadVar,
                     quantiles=[0.05,0.5,0.95],
                     start_date=pd.to_datetime('09-01-1958',format='%m-%d-%Y'),
                     averageWells=False,
                     minMax=False):
        """
        Process the outputs into a single DataFrame with specified percentiles
        """
        print(f'Processing {loadVar} Data....')
        ## The hydrus output calls the water flux "temp", I think this is specific 
        ## to the fact that I am using a Mac compiled version of the software...
        if loadVar == 'Flux_299':
            loadVar = 'Temp_299'
        mainDat = pd.DataFrame()
        ###############################################################################
        ## Making a single dataframe with data from each of the wells defined by nwells
        ###############################################################################
        for well in tqdm(self.nwells):
            tempDat = pd.DataFrame()
            ########################################################################
            ## Take each Monte Carlo realization and load it into a single dataframe
            ########################################################################
            for df_key in dataset[well].keys():
                df = dataset[well][df_key]
                tempDat = pd.concat([tempDat,df[loadVar]],axis=1)
            ##############################
            ## Take quantiles if specifeid
            ##############################
            if len(quantiles) > 0:
                ####################################################
                ## Take quantiles across the simulation realizations
                ####################################################
                quant = tempDat.quantile(quantiles,
                                         axis=1,
                                         numeric_only=True).transpose().head(51499)
                ## Replace the median quantile with the mean value
                ## Mostly doing this for simplicity when plotting
                quant[0.5] = tempDat.mean(axis=1,
                                          numeric_only=True).head(51499)
                ##################
                ## Convert to mg/L
                ##################
                if 'Conc' in loadVar:
                    quant[quantiles] = quant[quantiles] * 1000
                #################################
                ## Want to plot positive recharge
                #################################
                if loadVar == 'Temp_299':
                    quant[quantiles] = quant[quantiles] * -1
                ########################################################
                ## Rename the columns to match the well # they came from
                ########################################################
                quantDict = {}
                for q in quantiles:
                    quantDict[q] = f'{q}_{well}'
                quant = (quant.rename(quantDict,
                             axis=1)
                        .reset_index()
                        .drop('index',axis=1)
                        )
                ###############################################################################
                ## Concat that well onto the main dataset, or, return a well averaged DataFrame
                ###############################################################################
                if averageWells:
                    if len(mainDat) > 0:
                        mainDat = pd.DataFrame((quant.values+mainDat.values)/2,
                                                    columns=mainDat.columns,
                                                    index=mainDat.index) 
                    else:
                        mainDat = quant
                else:
                    mainDat = pd.concat([mainDat,quant],axis=1)
            ####################################################################################################
            ## Don't want quantiles, want the raw data, this allows for well and simulation averages to be taken
            ####################################################################################################
            else:
                ## Concat that well onto the main dataset
                tempDat.columns = [f'{x}_{well}' for x in tempDat.columns]
                mainDat = pd.concat([mainDat,tempDat],axis=1)
        ######################################################
        ## Set a datetime column based on the index (timestep)
        ######################################################
        mainDat['day'] = start_date + pd.to_timedelta(mainDat.index,unit='d')
        mainDat = mainDat.set_index('day')
        return mainDat
       
    
    def block_lookup(self):
        """
        Define a lookup table for the orchard blocks
        """
        nwells = ['NE1','NE1','NE1','SE','SE','NE2','NE2','NE2','SE','SE',
                 'NW','NW','SW1','SW1','SW2','NW','NW','SW1','SW1','SW2']
        blocks = dict(zip(list(range(1,21)), nwells))
        return blocks
                 
    
    def calc_stats_recharge(self):
        """
        Calculate CV stats for predicted recharge during 2012-2022
        """
        ## Calculate CV in predicted recharge across blocks in 2013-2022 GS's
        start = pd.to_datetime('08/31/2012')
        end = pd.to_datetime('09/01/2022')
        nodeinfo = self.clean_data(self.load_data(var='flux'))
        fluxData = self.process_data(dataset=nodeinfo,
                                     loadVar='Flux_299',
                                     quantiles=[])
        fluxData = fluxData[fluxData.index>start]
        fluxData = fluxData[fluxData.index<end]
        fluxData = fluxData.resample('12MS').sum()
        ## Set data for young trees to NaN
        NW = [16,11,17,12]
        for well in NW:
            fluxData.loc[fluxData.index<pd.to_datetime('09/01/2016'),f'Temp_299_{well}'] = np.NaN
        NE = [1,2,3,6,7,8]
        for well in NE:
            fluxData.loc[fluxData.index>pd.to_datetime('08/31/2021'),f'Temp_299_{well}'] = np.NaN
        SW1 = [18,13,19,14]
        for well in SW1:
            fluxData.loc[fluxData.index>pd.to_datetime('08/31/2017'),f'Temp_299_{well}'] = np.NaN
        ## Calculate CV values
        cvVals = []
        for key in fluxData.keys().unique():
            mean = abs(fluxData[key].mean(axis=1))
            std = abs(fluxData[key].std(axis=1))
            CV = std / mean
            CV = CV.mean()     
            cvVals.append(CV)
        meanCV = np.mean(cvVals)
        print(f'Average recharge across simulations for each well = {meanCV}')
        ## Calculate mean, std, and CV across wells
        mean = abs(fluxData.mean(axis=1))
        std = abs(fluxData.std(axis=1))
        CV = std / mean
        return(CV)
    
    
    def calc_stats_leaching(self):
        """
        Calculate CV stats for predicted leaching during 2012-2022
        """
        ## Calculate CV in predicted recharge across blocks in 2013-2022 GS's
        start = pd.to_datetime('08/31/2012')
        end = pd.to_datetime('09/01/2022')
        nodeinfo = self.clean_data(self.load_data(var='flux'))
        fluxData = self.process_data(dataset=nodeinfo,
                                     loadVar='Flux_299',
                                     quantiles=[])
        concData = self.process_data(dataset=nodeinfo,
                                     loadVar='Conc_299',
                                     quantiles=[]) / 1000
        leaching = pd.DataFrame(fluxData.values*concData.values,
                                    columns=fluxData.columns,
                                    index=fluxData.index) * 100
        leaching = leaching[leaching.index>start]
        leaching = leaching[leaching.index<end]
        leaching = leaching.resample('12MS').sum()
        ## Set data for young trees to NaN
        NW = [16,11,17,12]
        for well in NW:
            leaching.loc[leaching.index<pd.to_datetime('09/01/2016'),f'Temp_299_{well}'] = np.NaN
        NE = [1,2,3,6,7,8]
        for well in NE:
            leaching.loc[leaching.index>pd.to_datetime('08/31/2021'),f'Temp_299_{well}'] = np.NaN
        SW1 = [18,13,19,14]
        for well in SW1:
            leaching.loc[leaching.index>pd.to_datetime('08/31/2017'),f'Temp_299_{well}'] = np.NaN
        cvVals = []
        for key in leaching.keys().unique():
            mean = abs(leaching[key].mean(axis=1))
            std = abs(leaching[key].std(axis=1))
            CV = std / mean
            CV = CV.mean()     
            cvVals.append(CV)
        meanCV = np.mean(cvVals)
        print(f'Average CV of leaching across simulations for each well = {meanCV}')
        ## Calculate mean, std, and CV across wells
        mean = abs(leaching.mean(axis=1))
        std = abs(leaching.std(axis=1))
        CV = std / mean
        return(CV)
        
    
    def multi_panel(self,data_to_load='flux',load_var='Conc_299',
                         cumsum=False,
                         percentiles=True,
                         resample=False,
                         timeSlice=False,
                         quantiles=[0.05,0.5,0.95]):
        """
        Create multi-panel plot seperated by orchard block, defaults to nitrate
        leaching plot.
        
        data_to_load options:
            'flux' for obs_node.out file
            'solute' for solute1.out file
        
        load_var options:
            Conc_299 for solute concentration at bottom node
            Flux_299 for water flux at bottom node
            cvRoot for root solute uptake
            Sum(cvRoot) for cumulative root solute uptake
        """
        ########################
        ## Load the desired data
        ########################
        nodeinfo = self.clean_data(self.load_data(var=data_to_load))
        obs_data = self.process_data(dataset=nodeinfo,
                                     loadVar=load_var,
                                     quantiles=quantiles)
        blocks = self.block_lookup()
        ########################
        ## Initialize the figure
        ########################
        fig, ax = plt.subplots(3,2,figsize=[15,10])
        plot_idx = {'NE1':[0,0],'NE2':[0,1],'NW':[1,0],'SE':[1,1],
                    'SW1':[2,0],'SW2':[2,1]}
        try:
            fig.suptitle(self.title_dict[load_var],y=1.0,fontsize=15)
            fig.supylabel(self.ylabel_dict[load_var],x=0.0,fontsize=15)
        except:
            print('**** Add var to label/title dict ****')
        fig.supxlabel('Year',y=0.0,fontsize=15)
        fig.tight_layout(h_pad=1.5)
        ####################
        ## cumsum if desired
        ####################
        if cumsum:
            obs_data = obs_data.cumsum()
        if resample:
            obs_data = obs_data.resample('12MS').mean()
        if timeSlice:
            obs_data = obs_data[obs_data.index>pd.to_datetime('08/31/2012')]
            obs_data = obs_data[obs_data.index<pd.to_datetime('08/31/2022')]
        #######################################################
        ## Plot the average and percentiles for each simulation
        #######################################################
        for well in self.nwells:
            idx = plot_idx[blocks[well]]
            ## Ensuring that the colors are reset for each of the subplots
            color = next(ax[idx[0],idx[1]]._get_lines.prop_cycler)['color']
            ax[idx[0],idx[1]].plot(obs_data.index,
                                   obs_data[f'0.5_{well}'],
                                   label=f'MW {well}',
                                   color=color,
                                   ls='--')
            ## Give option to not plot the percentiles
            if percentiles:
                ## Sometimes errors out, this catches it
                try:
                    ax[idx[0],idx[1]].fill_between(obs_data.index,
                                                   obs_data[f'{quantiles[0]}_{well}'],
                                                   obs_data[f'{quantiles[2]}_{well}'],
                                                   alpha=0.5,
                                                   color=color)
                    ax[idx[0],idx[1]].plot(obs_data.index,
                                                   obs_data[f'{quantiles[0]}_{well}'],
                                                   color=color)
                    ax[idx[0],idx[1]].plot(obs_data.index,
                                                   obs_data[f'{quantiles[2]}_{well}'],
                                                   color=color)
                except:
                    print(f'*******Fill Between Failed for Well {well}*******')
            ax[idx[0],idx[1]].legend()
            ax[idx[0],idx[1]].set_title(blocks[well],loc='left') 
        
        
    def avg(self,quantiles=[0.05,0.5,0.95],dataset='flux',
                 load_var='Conc_299',
                 cumsum=False,
                 resample=False,
                 ax_limits=False):
        """
        Create the plot with a single average line, defaults to nitrate leaching 
        plot. Option to show cumulative average plot.
        
        dataset options:
            'flux' for obs_node.out file
            'solute' for solute1.out file
        
        load_var options:
            Conc_299 for solute concentration at bottom node
            Flux_299 for water flux at bottom node
            cvRoot for root solute uptake [mg cm-2 day-1]
            Sum(cvRoot) for cumulative root solute uptake
        """
        ## Load the desired data
        nodeinfo = self.clean_data(self.load_data(var=dataset))
        obs_data = self.process_data(dataset=nodeinfo,quantiles=[0.5],loadVar=load_var)
        ## Initialize the figure
        fig,ax = plt.subplots(figsize=[10,8])
        ax.set_xlabel('Year',fontsize=16)
        ## should add all variables into the dicts, but this catches the slack
        try:
            ax.set_title(f'Average {self.title_dict[load_var]}',fontsize=15)
            ax.set_ylabel(self.ylabel_dict[load_var],fontsize=16)
        except:
            ax.set_title(f'Orchard Average {load_var}',fontsize=15)
            ax.set_ylabel(load_var,fontsize=16)
        ## Take the quantiles across the wells to represent the orchard average values
        avgData = obs_data.quantile(quantiles,axis=1,interpolation='linear').transpose()
        avgData[0.5] = obs_data.mean(axis=1)
        if resample:
            avgData = avgData.resample('12MS').mean()
            print(avgData)
        ## If cumsum is True, plot cumsum data, otherwise plot original data
        if cumsum:
            ax.plot(avgData.index,avgData[0.5].cumsum(),color='red',linewidth=1)
            try:
                ax.fill_between(avgData.index,avgData[quantiles[0]].cumsum(),avgData[quantiles[2]].cumsum(),alpha=0.5,color='grey')
            except:
                print('fill_between not plotted')
            ax.set_title(f'Cumulative Average {self.title_dict[load_var]}',fontsize=15)
        else:
            ax.plot(avgData.index,avgData[0.5],
                    color='red',
                    linewidth=1,
                    label='Average Concentration [mg/L]')
            ax.axvline(pd.to_datetime('09/01/2017'),
                       ls='--',
                       color='black',
                       label='Switch to HFLC')
            try:
                ax.fill_between(avgData.index,avgData[quantiles[0]],
                                avgData[quantiles[2]],
                                alpha=0.5,
                                color='grey',
                                label='90% CL')
            except:
                print('fill_between not plotted')
            ax.legend(fontsize=12)
        if ax_limits:
            ax.set_xlim([pd.to_datetime('2013'),pd.to_datetime('2024')])
            
    
    def N_mass(self,plot=True,cumsum=False,
               percentiles=False,return_data=False):
        """
        Plot the mass of leached nitrate per block. With the default options, 
        this will calculate the leached mass and plot only the mean values for 
        each well.
        
        Options:
            plot: boolean to enable plotting or not. Implemented because this can
            be used a function for loading data only
            
            cumsum: Boolean to perfrorm a cumsum operation before plotting or 
            returning data
            
            percentiles: Boolean to control whether or not the 5th and 95th 
            fill-between percentiles are plotted
            
            return_data: Boolean to specify whether or not to return the mass_leached
            DataFrame.
        """
        nodeinfo = self.clean_data(self.load_data(var='flux'))
        ## Load nitrate concentration and water flux datasets
        leaching_conc = self.process_data(dataset=nodeinfo,loadVar='Conc_299')
        water_flux = self.process_data(dataset=nodeinfo,loadVar='Flux_299')
        water_flux = water_flux
        blocks = self.block_lookup()
        ## Doing a few conversions to get the results into kg-N/ha to compare with N mass balance
        ## liters to cm3
        leaching_conc = leaching_conc / 1000
        ## Calculating mass of leaching
        mass_leached = pd.DataFrame(water_flux.values*leaching_conc.values,
                                    columns=water_flux.columns,
                                    index=water_flux.index)
        ## kg/cm2 to kg/ha
        mass_leached = mass_leached * 100
        ## Plot cumsum if specified
        if cumsum:
            mass_leached = mass_leached.cumsum()
        if plot:
            ## Set up the figure
            fig, ax = plt.subplots(3,2,figsize=[13,10])
            plot_idx = {'NE1':[0,0],'NE2':[0,1],'NW':[1,0],'SE':[1,1],
                        'SW1':[2,0],'SW2':[2,1]}
            if cumsum:
                fig.suptitle('Cumulative N Mass/ha Leaching',y=0.94,fontsize=15)
            else:
                fig.suptitle('N Mass/ha Leaching',y=0.94,fontsize=15)
            fig.supylabel('kg/ha of nitrate',x=0.07,fontsize=15)
            fig.supxlabel('Year',y=0.06,fontsize=15)
            ## Plot the average and 95th percentiles for each simulation
            for well in self.nwells:
                idx = plot_idx[blocks[well]]
                ## Ensuring that the colors are reset for each of the subplots
                color = next(ax[idx[0],idx[1]]._get_lines.prop_cycler)['color']
                ax[idx[0],idx[1]].plot(mass_leached.index,
                                       mass_leached[f'0.5_{well}'],
                                       label=f'MW {well}',
                                       color=color)
                ## Only plot percentiles if specified
                if percentiles:
                    ax[idx[0],idx[1]].fill_between(mass_leached.index,
                                                   mass_leached[f'0.05_{well}'],
                                                   mass_leached[f'0.95_{well}'],
                                                   alpha=0.5,
                                                   color=color)
                ax[idx[0],idx[1]].legend()
                ax[idx[0],idx[1]].set_title(blocks[well])
        ## Return the mass_leached data, mostly for the N_balance_compare function
        if return_data:
            return mass_leached
    
    
    def precip_nitrate_contour(self,core=1):
        """
        Create the nitrate concentration contour + preciptiation gif
        Requires observation nodes every 10cm in the Hydrus simulations
        
        treeReplantings = {'NE1':[1976,2002,2022,2047,2072],
                                'NE2':[1970,1996,2022,2047,2072],
                                'NW':[1959,1985,2011,2037,2062,2087],
                                'SW1':[1976,2002,2018,2044,2069,2094],
                                'SW2':[1976,2002,2028,2053,2078],
                                'SE':[1958,1985,2010,2036,2061,2086]}
        """
        import matplotlib.ticker as ticker
        ## CHANGES AN ATTRIBUTE, MAKE SURE TO CHANGE BACK
        wells = self.nwells
        self.nwells = [core]
        ## Load the nitrate concentration data
        nodeinfo = self.clean_data(self.load_data(var='flux'))
        ## List of OBS nodes in HYDRUS
        node_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,
                     150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,299]
        ## Initialize the DataFrame
        contour_df = pd.DataFrame()
        for node in node_list:
            ## Pull 50th percentile (mean) for all the cores
            leaching_conc = self.process_data(dataset=nodeinfo,loadVar=f'Conc_{node}',quantiles=[0.5])
            ## Data processing function will automatically con
            contour_df[f'Conc_{node}'] = leaching_conc 
                
        ## Function to create contour output
        def f(x, y, contourArray):
            Z = np.array(np.zeros(Y.shape))
            for idx,row in enumerate(Y):
                if round(Y[idx][0]) == 300:
                    key = 299
                    val = contourArray[f'Conc_{key}']
                else:
                    key = round(Y[idx][0])
                    val = contourArray[f'Conc_{key}']
                for k,entry in enumerate(Z[idx]):
                    Z[idx][k] = val
            return Z

        ## Interval of timesteps to plot at
        plot_vals = [x*30 for x in range(1716)]
        plot_vals = plot_vals[1:]
        ## Load NE1 Precip data
        NE1_P = pd.read_pickle('./NE1_precip')
        ## Concentration at bottom of model (GW recharge)
        Conc_299 = contour_df['Conc_299']

        ## Create a plot for every value in plot_vals to be turned into a gif later
        print('**** Creating plots, this will take a while ****')
        for i in tqdm(plot_vals):
            i = int(i)
            ## Nitrate concentration data
            contourArray = contour_df.iloc[i]
            ## x and y arrays, y corresponds to 1D model height and x is arbitrary
            y = np.linspace(10,300,30)
            x = np.linspace(0,1,5)
            ## Turn x and y into grid objects for the contour
            X,Y = np.meshgrid(x,y)
            ## Contour data based on X and Y
            Z = f(X, Y, contourArray)
            ## Start date of the simulation
            startDate = pd.to_datetime('09/01/1958',format='%m/%d/%Y')
            ## Current date and a date label
            date = startDate + pd.to_timedelta(i,unit='d')
            date_label = date.strftime('%b %Y')
            ## NE1
            replants = [1976,2002,2022,2047,2072]
            ## SW1
            replants = [1976,2002,2018,2044,2069,2094]
            ###########################################################################
            #### Create the mosaic subplot
            ###########################################################################
            x = [['A panel', 'B panel',],
                 ['C panel','C panel']]
            gs_kw = dict(width_ratios=[1, 3.0], height_ratios=[3.0, 1])
            fig, ax = plt.subplot_mosaic(x,figsize=(10, 8), layout="constrained",
                                          gridspec_kw=gs_kw)
            ## Plot sup-title
            fig.suptitle(f'Well 1 (NE1) Nitrate Concentration and Precipitation: {date_label}',
                         fontsize=14)
            ###########################################################################
            #### Create the contour
            ###########################################################################
            contour = ax['A panel'].imshow(Z,extent=[0,70,-300,0],vmin=0,vmax=100)
            ## Add a colorbar
            plt.colorbar(contour,ax=ax['A panel'],
                         label='NO3-N Concentration [mg/L]',
                         )
            ## Get rid of xticks and set a y_label
            ax['A panel'].set_xticks([])
            ax['A panel'].set_ylabel('Depth [cm]',
                                     fontsize=13)
            ###########################################################################
            #### Plotting past 30-days moving precipitation + irrigation 
            ###########################################################################
            startWindow = date - pd.to_timedelta(30,unit='d')
            ## Get 30-day slice of precip data
            precip = NE1_P.loc[(NE1_P.index<=date)&(NE1_P.index>startWindow)]
            #precip = NE1_P.loc[NE1_P.index<=date] / 10 / 10
            ## Set a title and plot precip data
            ax['B panel'].set_title('Past 30-day P+I [cm]',fontsize=13)
            ax['B panel'].bar(precip.index,precip)
            ## Want to set a ylim
            ax['B panel'].set_ylim([0,6])
            tick_spacing = 10
            ax['B panel'].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ## If a replant year, color the plot background red
            if date.year in replants:
                pass
            ###########################################################################
            #### Plotting the recharge concentration curve as it's built
            ###########################################################################
            conc_slice = Conc_299.loc[Conc_299.index<=date]
            ax['C panel'].plot(conc_slice.index,conc_slice)
            ax['C panel'].ticklabel_format(axis='y',useOffset=False)
            ax['C panel'].set_ylim([0,120])
            ax['C panel'].set_title('Long-Term Nitrate Leaching Concentration')
            ax['C panel'].set_ylabel('Concentration [mg/L]')
            #plt.show()
            ## Save the figure, I think plt.close saves memory...
            plt.savefig(f'/Users/spencerjordan/Documents/Hydrus/python_scripts/nitrate_precip_gif/image_{i}.png')
            plt.close()
        ## Change nwells back
        self.nwells = wells
        ## Create the .gif file
        self.png_to_gif()
        
        
    def png_to_gif(self):
        """
        Process PNG files into a single .gif
        """
        print('**** Creating .gif file ****')
        import glob
        from PIL import Image
        import os
        # Directory where images to be turned into a gif are held 
        img_DIR = '/Users/spencerjordan/Documents/Hydrus/python_scripts/nitrate_precip_gif/'
        ## Directory and NAME of gif
        save_DIR = '/Users/spencerjordan/Documents/Hydrus/python_scripts/nitrate_precip_gif/nitratePrecip.gif'
        # Create a list of frames from the images in the gif
        frames = []
        imgs = glob.glob(f'{img_DIR}image_*.png')
        ## Sort by creation time
        imgs.sort(key=os.path.getmtime)

        for i,image in enumerate(imgs):
            ## For leaching gif
            #idx = (i) * 50 + 1
            ## For better x-section leaching gif
            #idx = 12 * i
            new_frame = Image.open(image)
            frames.append(new_frame)
        # Save into a GIF file that loops forever
        # Adjust the speed of image chages with duration --> Should be in ms
        frames[0].save(save_DIR, format='GIF',append_images=frames[1:],save_all=True,
                       duration=200, loop=0)
    
    
    def organizeBlocks(self,dataset):
        """
        Function to take the average of well data across each orchard block
        """
        ## Blocks to match the mass balance and wells
        blocks = {'NE':[1,2,3,6,7,8],
                  'NW':[12,11,17,16],
                  'SE':[4,5,9,10],
                  'SW':[13,14,15,18,19,20]}
        block_data = pd.DataFrame()
        for block in blocks:
            ## Average across each well in the block
            block_avg = pd.DataFrame()
            for well in blocks[block]:
                if well in self.nwells:
                    temp = dataset[f'0.5_{well}']
                    block_avg = pd.concat([block_avg,temp],axis=1)
                else:
                    pass
            ## Take the average of the wells
            block_avg = block_avg.mean(axis=1)
            ## Rename to the block of the orchard
            block_avg.name = f'{block}'
            ## Concat into the DataFrame --> This is the block average
            block_data = pd.concat([block_data,block_avg],axis=1)
        return block_data
    
        
    def N_balance_bars(self):
        """
        Make a direct comparison to the mass balance with denit, mineralization, 
        depostion, crop uptake, and leaching
        """
        #####################
        ## Load the Data
        #####################
        ## Load solute data --> undo cumulative sum, resample to water-years, and take the sum
        solute_data = self.clean_data(self.load_data(var='solute'))
        ## Actual solute flux across the soil surface
        cvTop = self.process_data(quantiles=[0.5],dataset=solute_data,loadVar='Sum(cvTop)').diff().resample('12MS').sum()
        ## Solute added to the flow region by zero-order reactions
        cvCh0 = self.process_data(quantiles=[0.5],dataset=solute_data,loadVar='cvCh0').diff().resample('12MS').sum() #--> Mineralization
        ## Solute removed from the flow region by first-order reactions
        cvCh1 = self.process_data(quantiles=[0.5],dataset=solute_data,loadVar='cvCh1').diff().resample('12MS').sum() #--> Denitrification
        ## Cumulative amount of solute removed from the flow region by root water uptake S
        cvRoot = self.process_data(quantiles=[0.5],dataset=solute_data,loadVar='Sum(cvRoot)').diff().resample('12MS').sum()
        ## Leaching Concentration at bottom of model --> concentration * flux
        nodeinfo = self.clean_data(self.load_data(var='flux'))
        ## My functions output mg/L, but here want to compute mg/cm3
        conc_299 = self.process_data(dataset=nodeinfo,quantiles=[0.5],loadVar='Conc_299') / 1000
        flux_299 = self.process_data(dataset=nodeinfo,quantiles=[0.5],loadVar='Flux_299')
        leaching = pd.DataFrame(flux_299.values*conc_299.values,
                                    columns=flux_299.columns,
                                    index=flux_299.index).resample('12MS').sum()
        ## Calc leaching for the 15 years pre and post HFLC
        preHFLC = leaching[(leaching.index>pd.to_datetime('08/31/2001'))&(leaching.index<pd.to_datetime('09/01/2017'))].mean(axis=1).mean()
        print(f'pre leaching: {preHFLC}')
        postHFLC = leaching[(leaching.index>pd.to_datetime('08/31/2070'))&(leaching.index<pd.to_datetime('09/01/2090'))].mean(axis=1).mean()
        print(f'post leaching: {postHFLC}')

        #####################
        ## Process the data
        #####################
        ## Get average mass-leaching values based on the four blocks, slice data to timeframe, and resample by water year
        start = pd.to_datetime('08/31/2012')
        end = pd.to_datetime('09/01/2022')
        cvTop = self.organizeBlocks(cvTop)
        fert = cvTop.set_index(pd.to_datetime(cvTop.index))[(cvTop.index>start)&(cvTop.index<end)]
        cvCh0 = self.organizeBlocks(cvCh0)
        mineral = cvCh0.set_index(pd.to_datetime(cvCh0.index))[(cvCh0.index>start)&(cvCh0.index<end)]
        cvCh1 = self.organizeBlocks(cvCh1)
        denit = cvCh1.set_index(pd.to_datetime(cvCh1.index))[(cvCh1.index>start)&(cvCh1.index<end)]
        cvRoot = self.organizeBlocks(cvRoot)
        treeUptake = cvRoot.set_index(pd.to_datetime(cvRoot.index))[(cvRoot.index>start)&(cvRoot.index<end)]
        leaching = self.organizeBlocks(leaching)
        leaching = leaching.set_index(pd.to_datetime(leaching.index))[(leaching.index>start)&(leaching.index<end)]
        ######################
        ## Create the Figure
        ######################
        GS = [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
        ## Create the plot object, 3 rows with 9 columns
        fig, ax = plt.subplots(3,10,figsize=(14,10))
        ## Add a title and xlabel applied to the whole figure, adjusting title position with 'y'
        fig.suptitle('Modeled Annual N Fluxes and Leaching at the Water Table (7m Depth)',fontsize=22,y=1.005)
        fig.supxlabel('Orchard Plot',fontsize=16)
        ## Helps with layout formatting, adding verticle and horitoztal padding
        fig.tight_layout(w_pad=-2,h_pad=2)
        #####################
        #### Inputs plot
        #####################
        for idx,year in enumerate(GS):
            mineralSub = mineral[mineral.index.year==(year-1)].T * 100
            mineralSub.columns = ['mineral']
            fertSub = fert[fert.index.year==(year-1)].T * 100
            fertSub.columns = ['fert']
            inputs = pd.concat([mineralSub,fertSub],axis=1)
            inputs.plot(ax=ax[0,idx],kind='bar',stacked=True,
                        legend=False,
                        width=0.8)
            ax[0,idx].set_ylim([0,400])
            ax[0,idx].set_title(year)
            # Turn off y-tick labels for all plots except the first
            if idx:
                ax[0,idx].set_yticklabels([])
        ## Set the axis label and legend
        ax[0,0].set_ylabel('N Inputs [kg/ha]',fontsize=16)
        ## Add a legend based on data from the last panel
        ax[0,9].legend(loc=5,bbox_to_anchor=(1.68, 0.3, 0.5, 0.5),fontsize=13)
        #####################
        #### Outputs plot
        #####################
        for idx,year in enumerate(GS):
            treeSub = treeUptake[treeUptake.index.year==(year-1)].T * -100
            treeSub.columns = ['crop']
            denitSub = denit[denit.index.year==(year-1)].T * 100
            denitSub.columns = ['denit']
            outputs = pd.concat([denitSub,treeSub],axis=1)
            outputs.plot(ax=ax[1,idx],kind='bar',stacked=True,
                         legend=False,
                         width=0.8)
            ax[1,idx].set_ylim([-350,0])
            # Turn off y-tick labels for all plots except the first
            if idx:
                ax[1,idx].set_yticklabels([])
        ## Set the axis label and legend
        ax[1,0].set_ylabel('N Outputs [kg/ha]',fontsize=16)
        ## Add a legend based on data from the last panel
        ax[1,9].legend(loc=5,bbox_to_anchor=(1.55, 0.3, 0.5, 0.5),fontsize=13)
        #####################
        #### Leaching plot
        #####################
        for idx,year in enumerate(GS):
            ## Convert from mg/cm3 to kg/ha, and want a negative value
            leachingSub = leaching[leaching.index.year==(year-1)].T * (-100)
            leachingSub.columns = ['leaching']
            leachingSub.plot(ax=ax[2,idx],kind='bar',
                             legend=False,
                             width=0.8)
            ax[2,idx].set_ylim([-300,0])
            ## Turn off y-tick labels for all plots except the first
            if idx:
                ax[2,idx].set_yticklabels([])
        ## Set the axis label and legend
        ax[2,0].set_ylabel('Leaching [kg/ha]',fontsize=16)
        print(f'Average Leaching == {leaching.mean(axis=1).mean()}')
            
        
    def NUE_hist(self,remove_replants=False):
        """
        Plot historgrams of modeled N efficiency before and after the switch to HFLC
        N_efficiency = (root_uptake + denitrification)/(fert + atm_dep + mineralization)
        
        Look at splitting between 2017 and splitting between hflc/non-hflc
        type leaching years
        
        """
        import scipy.stats as stats
        ## Load solute data --> undo cumulative sum, resample to water-years, and take the sum
        solute_data = self.clean_data(self.load_data(var='solute'))
        ## Actual solute flux across the soil surface
        cvTop = self.process_data(quantiles=[],dataset=solute_data,loadVar='Sum(cvTop)').diff().resample('12MS').sum()
        ## Solute added to the flow region by zero-order reactions
        cvCh0 = self.process_data(quantiles=[],dataset=solute_data,loadVar='cvCh0').diff().resample('12MS').sum() #--> Mineralization
        ## Solute removed from the flow region by first-order reactions
        cvCh1 = self.process_data(quantiles=[],dataset=solute_data,loadVar='cvCh1').diff().resample('12MS').sum() #--> Denitrification
        ## Cumulative amount of solute removed from the flow region by root water uptake S
        cvRoot = self.process_data(quantiles=[],dataset=solute_data,loadVar='Sum(cvRoot)').diff().resample('12MS').sum()
        ## Calculate modeled NUE
        modeledNUE = pd.DataFrame((-cvCh1.values+cvRoot.values)/(cvTop.values+cvCh0.values),
                                    columns=cvTop.columns,
                                    index=cvTop.index)
        modeledNUE_avg = modeledNUE.copy().quantile(q=[0.1,0.5,0.9],axis=1).T
        modeledNUE_avg[0.5] = modeledNUE.mean(axis=1)

        ## If specified, remove the data that includes immature trees
        if remove_replants:
            #####################################
            ## Drop all years with immature trees
            #####################################
            treeReplantings = {'NE1':[1976,2002,2022,2047,2072],
                                    'NE2':[1970,1996,2022,2047,2072],
                                    'NW':[1959,1985,2011,2037,2062,2087],
                                    'SW1':[1976,2002,2018,2044,2069,2094],
                                    'SW2':[1976,2002,2028,2053,2078],
                                    'SE':[1958,1985,2010,2036,2061,2086]}
            blocks = {'NE1':[1,2,3],
                      'NE2':[6,7,8],
                      'NW':[12,11,17,16],
                      'SE':[4,5,9,10],
                      'SW1':[13,14,18,19],
                      'SW2':[15,20]}
            for block in treeReplantings.keys():
                wells = blocks[block]
                wellKey = [f'Sum(cvTop)_{well}' for well in wells]
                for year in treeReplantings[block]:
                    for k in range(6):
                        ## Set to NaN so they are ignored in the calculations
                        modeledNUE.loc[modeledNUE.index==pd.to_datetime(f'09/01/{year-1+k}'),wellKey] = np.NaN
        
        ###############        
        ## Pre HFLC NUE --> slice data and then stack into a single numpy array 
        ###############
        preNUE = np.hstack(modeledNUE[modeledNUE.index<pd.to_datetime('09/01/2017')].values)
        print(f'data length pre = {len(preNUE)}')
        print(f'preHFLC: min = {min(preNUE)}, max = {max(preNUE)}, median = {np.nanmedian(preNUE)}')
        preNUE_avg = modeledNUE_avg[modeledNUE_avg.index<pd.to_datetime('09/01/2017')]
        print('mean + cl =',preNUE_avg[-5:])
        preNUE = preNUE[preNUE>0]
        ################
        ## Post HFLC NUE
        ################
        postNUE = np.hstack(modeledNUE[modeledNUE.index>=pd.to_datetime('09/01/2017')].values)
        #postNUE = postNUE[postNUE>0]
        print(f'data length post = {len(postNUE)}')
        print(f'postHFLC: min = {min(postNUE)}, max = {max(postNUE)}, median = {np.nanmedian(postNUE)}')
        postNUE_avg = modeledNUE_avg[modeledNUE_avg.index>=pd.to_datetime('09/01/2017')]
        print('mean + cl =',postNUE_avg[:5])
        postNUE = postNUE[postNUE>0]
        #####################
        ## Perform the t-test
        #####################
        print(stats.ttest_ind(postNUE, preNUE, equal_var = False))
        ## Initialize the figure
        ## Some stuff for the title formatting
        fig,ax = plt.subplots(2,1,figsize=[6,4])
        fig.tight_layout()
        fig.suptitle('Normalized Hstrogram of Modeled Annual N Efficiency',
                     y=1.06,
                     fontsize=12)
        fig.supylabel('Probability',fontsize=12,x=-0.01)
        fig.supxlabel('N use efficiency',y=-0.02,fontsize=12)
        ## Pre
        weights = np.ones_like(preNUE) / len(preNUE)
        ax[0].set_title('NUE = (root uptake + denitrification)/(fert + atm. dep. + mineralization)',
                        fontsize=10)
        ax[0].hist(preNUE,
                   bins=9,
                   edgecolor='black',
                   color='coral',
                   weights=weights,
                   label='Pre-HFLC: 1958-2018')
        ax[0].axvline(np.nanmedian(preNUE),ls='--',color='red',lw=2,
                      label=f'HFLC: {round(np.nanmedian(preNUE),2)}')
        ax[0].set_xlim([0,1.6])
        ax[0].set_ylim([0,0.5])
        ax[0].legend(fontsize=9)
        ## Post
        weights = np.ones_like(postNUE) / len(postNUE)
        ax[1].hist(postNUE,
                   bins=10,
                   edgecolor='black',
                   color='cornflowerblue',
                   weights=weights,
                   label='HFLC: 2018-2100')
        ax[1].axvline(np.nanmedian(postNUE),ls='--',color='blue',lw=2,
                      label=f'HFLC: {round(np.nanmedian(postNUE),2)}')
        ax[1].set_xlim([0,1.6])
        ax[1].set_ylim([0,0.5])
        ax[1].legend(fontsize=9)
        return preNUE,postNUE


    def monthly_leaching(self):
        """
        Plot to show the monthly leaching for the 15 years before and after the switch to HFLC
        """
        quantiles = [0.05,0.5,0.95]
        nodeinfo = self.clean_data(self.load_data(var='flux'))
        conc_299 = self.process_data(dataset=nodeinfo,quantiles=quantiles,loadVar='Conc_299',averageWells=True)
        conc_299 = conc_299 / 1000
        flux_299 = self.process_data(dataset=nodeinfo,quantiles=quantiles,loadVar='Flux_299',averageWells=True)
        leachingMass = pd.DataFrame(conc_299.values*flux_299.values,
                                    columns=flux_299.columns,
                                    index=flux_299.index).resample('1M').sum()
        ## kg/cm2 to kg/ha
        leachingMass = leachingMass * 100
        ## Leaching for 15 years before and after the switch to HFLC
        preHFLC = leachingMass[(leachingMass.index<pd.to_datetime('09/01/2010'))&(leachingMass.index>pd.to_datetime('08/31/1970'))]
        postHFLC = leachingMass[(leachingMass.index>pd.to_datetime('08/31/2050'))&(leachingMass.index<pd.to_datetime('09/01/2090'))]
        ## Groupby month
        preHFLC = preHFLC.groupby(preHFLC.index.month).mean()
        postHFLC = postHFLC.groupby(postHFLC.index.month).mean()
        ## Calculate the percent decrease post-hflc
        percentReduction = (preHFLC['0.5_1']-postHFLC['0.5_1'])/preHFLC['0.5_1'] * 100
        ## Initialize the mosaic plot
        fig, ax = plt.subplot_mosaic([['upper left', 'upper right'],
                                      ['upper left', 'upper right'],
                                      ['lower', 'lower']],
                                     figsize=(8,8), layout="constrained")
        fig.suptitle('Modeled N Leaching Rates\n20 soil profiles: 1970-2010 and 2050-2090',
                     fontsize=12)
        ## Pre HFLC
        ax['upper left'].set_title('Pre-HFLC')
        ax['upper left'].plot(preHFLC['0.5_1'],color='red')
        ax['upper left'].fill_between(preHFLC.index,preHFLC[f'{quantiles[0]}_1'],preHFLC[f'{quantiles[2]}_1'],
                                      alpha=0.5)
        ax['upper left'].set_ylim([0,10])
        ## Set the x-ticks
        ax['upper left'].set_xticks(np.arange(len(preHFLC))+1)
        labelNew = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        labels = [item.get_text() for item in ax['upper left'].get_xticklabels()]
        for idx,k in enumerate(labels):
            labels[idx] = labelNew[idx]
        ax['upper left'].set_xticklabels(labels,rotation=60)
        ax['upper left'].set_ylabel('kg/ha/month',fontsize=12)
        ## Post HFLC
        ax['upper right'].set_title('HFLC')
        ax['upper right'].plot(postHFLC['0.5_1'],color='red',label='Mean')
        ax['upper right'].fill_between(postHFLC.index,postHFLC[f'{quantiles[0]}_1'],postHFLC[f'{quantiles[2]}_1'],
                                      alpha=0.5,label='90% Cl')
        ax['upper right'].legend(fontsize=12)
        ax['upper right'].set_ylim([0,10])
        ## Set the x-ticks
        ax['upper right'].set_xticks(np.arange(len(preHFLC))+1)
        labels = [item.get_text() for item in ax['upper right'].get_xticklabels()]
        for idx,k in enumerate(labels):
            labels[idx] = labelNew[idx]
        ax['upper right'].set_xticklabels(labels,rotation=60)
        ## Percent reduction plot
        ax['lower'].bar(x=percentReduction.index,
                        height=percentReduction,
                        color='coral',
                        edgecolor='black',
                        linewidth=0.8)
        ax['lower'].set_ylabel('Percent Reduction',
                               fontsize=12)
        ax['lower'].set_title('Average reduction in monthly leaching rate after switching to HFLC',
                              fontsize=12)
        ax['lower'].set_xlabel('Month',
                               fontsize=12)
        ## Set the x-ticks
        ax['lower'].set_xticks(np.arange(len(preHFLC))+1)
        labels = [item.get_text() for item in ax['lower'].get_xticklabels()]
        for idx,k in enumerate(labels):
            labels[idx] = labelNew[idx]
        ax['lower'].set_xticklabels(labels,rotation=60)
        
        
    def gw_recharge_compare(self,cumsum=False):
        nodeinfo = self.load_data(var='flux')
        nodeinfo = self.clean_data(nodeinfo)
        hydrus_recharge = self.process_data(quantiles=[0.5],dataset=nodeinfo,loadVar='Flux_299')
        hydrus_recharge = hydrus_recharge.resample('1m').sum()
        ## Load data pickled from waterBalance.py that calculates orchard water balance
        rch = pd.read_pickle('/Users/spencerjordan/Documents/Hydrus/python_scripts/massBalance_recharge.p')
        rch = rch[rch.index<pd.to_datetime('09/01/2022')]
        ## Cut down the Hydrus results to only the time period the mass balance has data for
        startDate = rch.index.min()
        endDate = rch.index.max()
        hydrus_recharge = hydrus_recharge.loc[(hydrus_recharge.index>=startDate)&
                                              (hydrus_recharge.index<=endDate),:]
        ## Mean recharge across orchard
        hydrus_recharge = hydrus_recharge.mean(axis=1)
        ## Option for cumulative sum plots
        if cumsum:
            hydrus_recharge = hydrus_recharge.cumsum()
            #rch.loc[rch['recharge']<0,'recharge'] = 0
            rch = rch.cumsum()
        ## Initialize the figure
        fig,ax1 = plt.subplots(figsize=[10,7])
        ## Plot Hydrus predicted recharge
        ax1.bar(hydrus_recharge.index,hydrus_recharge.values,color='red',alpha=0.5,
                width=24,label='Simulated Recharge')
        ## Plot the mass balance predicted recharge
        ax1.bar(rch.index,rch['avgBalance'],label='Mass Balance Recharge',
                width=24,color='blue',alpha=0.5)
        ax1.axhline(0,c='black',ls='--')
        if cumsum:
            ax1.set_title('Cumulative Orchard Average GW Recharge: Modeled and Mass Balance',
                          fontsize=14)
        else:
            ax1.set_title('Orchard Average GW Recharge: Modeled and Mass Balance',
                          fontsize=14)
        ## Set the upper xlim to improve plot aesthetics
        ax1.set_xlim(right=rch.index.max())
        ## Don't want the ylim if using the cumsum plots and adding a note about
        ## the mass balance recharge values if using cumsum
        if cumsum:
            ax1.text(0.02, 0.8, '**Negative recharge values dropped\n   from mass balance', horizontalalignment='left',
                     verticalalignment='center', transform=ax1.transAxes,fontsize=11)
        else:
            #ax1.set_ylim([-10,10])
            pass
        ax1.set_xlabel('Date',fontsize=14)
        ax1.set_ylabel('Recharge [cm]',fontsize=14)
        ax1.legend(fontsize=12)
        plt.xticks(rotation=60, fontsize=10)
    
        
    def root_uptake_compare(self,save=False):
        """
        Create comparison plot between measured and modeled N root uptake
        """
        ## Load the root uptake data from the hydrus results
        nodeinfo = self.load_data(var='solute')
        nodeinfo = self.clean_data(nodeinfo)
        obs_data = self.process_data(dataset=nodeinfo,loadVar='cvRoot')
        blocks = self.block_lookup()
        ## Initialize the multi-panel figure
        fig, ax = plt.subplots(3,2,figsize=[13,10])
        fig.suptitle('Measured Root Uptake and Modeled sum(cvRoot)',fontsize=14,y=0.935)
        ## Load the measured root solute uptake data
        uptake_measured = pd.read_csv('/Users/spencerjordan/Documents/bowman_data_analysis/N_mass_balance/manual_mass_balance_2022.csv')
        ## Convert mg/cm2 to kg/ha by dividing by 100
        uptake_measured['Uptake mg/cm2'] = (uptake_measured['Growth'] + uptake_measured['Uptake kg/ha']) / 100
        ## Inlcuding 09/30 in the GS column to get proper placement of data points
        for u,year in enumerate(uptake_measured['GS']):
            uptake_measured['GS'][u] = f'{year}/09/30' 
        ## Set datetime index
        uptake_measured = uptake_measured.set_index('GS')
        uptake_measured.index = pd.to_datetime(uptake_measured.index,format='%Y/%m/%d')
        plot_idx = {'NE1':[0,0],'NE2':[0,1],'NW':[1,0],'SE':[1,1],
                    'SW1':[2,0],'SW2':[2,1]}
        ## Cut the hydrus data to the span of the measured root uptake data
        obs_data = obs_data[obs_data.index>pd.to_datetime('2012/10/01',format='%Y/%m/%d')]
        obs_data = obs_data[obs_data.index<pd.to_datetime('2022/10/01',format='%Y/%m/%d')]
        ## Plot the measured uptake
        for well in self.nwells:
            ## Plotting index in the subplot
            idx = plot_idx[blocks[well]]
            ## Reset color cycling for each panel
            color = next(ax[idx[0],idx[1]]._get_lines.prop_cycler)['color']
            ## Plot HYDRUS data
            ax[idx[0],idx[1]].plot(obs_data.index,
                                   obs_data[f'0.5_{well}'].cumsum(),
                                   label=f'MW {well}',
                                   color=color)
            ax[idx[0],idx[1]].fill_between(obs_data.index,
                                           obs_data[f'0.05_{well}'].cumsum(),
                                           obs_data[f'0.95_{well}'].cumsum(),
                                           alpha=0.5,
                                           color=color)
            ## Plot measured uptake
            uptake_measured.loc[uptake_measured['block']==blocks[well][0:2],'Uptake mg/cm2'].cumsum().plot(ax=ax[idx[0],idx[1]],label=f'{well} measured')
            ax[idx[0],idx[1]].legend()
            ax[idx[0],idx[1]].set_title(blocks[well])
            ax[idx[0],idx[1]].set_xlim([pd.to_datetime('2012/10/01',format='%Y/%m/%d'),
                         pd.to_datetime('2022/10/01',format='%Y/%m/%d')])
        ## Save the figure if save is True
        if save:
            plt.savefig('/Users/spencerjordan/Documents/Hydrus/python_scripts/plots/N_uptake_compare.png',dpi=200)
    

    def ET_compare(self):
        """
        plot potential (inputted in ATMOSPH.IN file) vs actual ET in the simulation
        """
        ## Load the ET inputs for comparison
        a = atmosph()
        a.main()
        ## Load the tlevel data
        tlevelinfo = self.load_data(var='tlevel')
        tlevelinfo = self.clean_data(tlevelinfo)
        ## Load transpiration variable
        rRoot = self.process_data(dataset=tlevelinfo,loadVar='rRoot',
                                  quantiles=[0.5])
        ## Taking cumulative sum
        rRoot = rRoot.cumsum()
        ## Load evaporation variable (already in cumsum form)
        rTop = self.process_data(dataset=tlevelinfo,loadVar='sum(Evap)',
                                 quantiles=[0.5])
        ## Calculate total modeled ET
        ET_modeled = pd.DataFrame(rRoot.values+rTop.values,
                                    columns=rRoot.columns,
                                    index=rRoot.index)
        ## Initialize the figure
        fig,axs = plt.subplots(3,2,figsize=[13,10])
        fig.supylabel('Cumulative ET [cm]',x=0.05,fontsize=15)
        fig.supxlabel('Time',y=0.06,fontsize=15)
        fig.suptitle('Input vs Modeled ET by Orchard Block',y=0.94,fontsize=15)
        ## Relate plots to simulation indexes
        lookup_dict = {'NE1_ET':['0.5_1','0.5_2','0.5_3'],
                       'NE2_ET':['0.5_6','0.5_7','0.5_8'],
                       'NW_ET':['0.5_11','0.5_12','0.5_16','0.5_17'],
                       'SE_ET':['0.5_4','0.5_5','0.5_10','0.5_9'],
                       'SW1_ET':['0.5_13','0.5_14','0.5_18','0.5_19'],
                       'SW2_ET':['0.5_15','0.5_20']}
        for block, ax in zip(lookup_dict.keys(),axs.ravel()):
            modeled = ET_modeled[lookup_dict[block]].mean(axis=1)
            a.atmosph_data[block].cumsum().plot(ax=ax,label='Input ET',
                                                color='blue',lw=2)
            modeled.plot(ax=ax,label='Modeled ET',ls='-.',
                         color='red',lw=2)
            title = block.split('_')[0]
            ax.set_title(f'{title}',y=0.99)
            ## Set one legend for entire block
            if block == 'NE1_ET':
                ax.legend()
            percent_diff = abs(round((modeled.max()-a.atmosph_data['NE1_ET'].cumsum().max())/a.atmosph_data['NE1_ET'].cumsum().max(),2) * 100)
            ax.text(0.1,0.6,f'percent error = {percent_diff}%',transform=ax.transAxes,
                    fontsize=12)
            ax.set_xlabel('')
        
        
    def monthly_ETa_compare(self):
        """
        Compared inputted to actual ETa values
        """
        a = atmosph()
        a.main()
        ## Load the tlevel data
        tlevelinfo = self.load_data(var='tlevel')
        tlevelinfo = self.clean_data(tlevelinfo)
        ## Load transpiration variable
        rRoot = self.process_data(dataset=tlevelinfo,loadVar='rRoot',
                                  quantiles=[0.5])
        ## Load evaporation variable (already in cumsum form)
        rTop = self.process_data(dataset=tlevelinfo,loadVar='sum(Evap)',
                                 quantiles=[0.5]).diff()
        ## Calculate total modeled ET
        ET_modeled = pd.DataFrame(rRoot.values+rTop.values,
                                    columns=rRoot.columns,
                                    index=rRoot.index) 
        ET_modeled = ET_modeled.resample('1M').sum()
        ET_modeled = ET_modeled[ET_modeled.index>pd.to_datetime('08/31/2012')]
        ET_modeled = ET_modeled[ET_modeled.index<pd.to_datetime('09/01/2022')]

        ## Initialize the figure
        fig,axs = plt.subplots(6,1,figsize=[13,10])
        fig.supylabel('Monthly ET [cm]',fontsize=15)
        fig.supxlabel('Year',fontsize=15)
        fig.suptitle('Monthly Input ETc vs Modeled ETa by Orchard Block',fontsize=15)
        fig.tight_layout()
        ## Relate plots to simulation indexes
        lookup_dict = {'NE1_ET':['0.5_1','0.5_2','0.5_3'],
                       'NE2_ET':['0.5_6','0.5_7','0.5_8'],
                       'NW_ET':['0.5_11','0.5_12','0.5_16','0.5_17'],
                       'SE_ET':['0.5_4','0.5_5','0.5_10','0.5_9'],
                       'SW1_ET':['0.5_13','0.5_14','0.5_18','0.5_19'],
                       'SW2_ET':['0.5_15','0.5_20']}
        for block, ax in zip(lookup_dict.keys(),axs.ravel()):
            modeled = ET_modeled[lookup_dict[block]].mean(axis=1)
            measured = a.atmosph_data[block].resample('1M').sum()
            measured = measured[measured.index>pd.to_datetime('08/31/2012')]
            measured = measured[measured.index<pd.to_datetime('09/01/2022')]
            #measured.plot(ax=ax,
            #              label='Input ETc',
            #              color='blue',
            #              kind='bar')
            ax.bar(measured.index,
                   measured.values,
                   width=20,
                   label='Input ETc')
            #ax.bar(modeled.index,
            #       modeled.values,
            #       width=15,
            #       alpha=0.5)
            ax.plot(modeled.index,
                    modeled.values,
                    color='orange',
                    label='Modeled ETa')
            #modeled.plot(ax=ax,
            #             label='Modeled ETa',
            #             color='red',
            #             kind='bar',
            #             alpha=0.7)
            title = block.split('_')[0]
            ax.set_title(f'{title}',
                         y=0.99,
                         loc='left')
            #ax.set_xlim([modeled.index.min(),modeled.index.max()])
            ## Set one legend for entire block
            if block == 'NE1_ET':
                ax.legend()
            ax.set_xlabel('')
            
        
    def annual_modeled_recharge(self):
        """
        Plot the modeled annual average recharge per orchard block
        """
        nodeinfo = self.load_data(var='flux')
        nodeinfo = self.clean_data(nodeinfo)
        ## Load the modeled recharge
        hydrus_recharge = self.process_data(quantiles=[0.5],dataset=nodeinfo,loadVar='Flux_299')
        ## Slice the data to show only GSs 2013 --> 2022
        hydrus_recharge = hydrus_recharge[hydrus_recharge.index>pd.to_datetime('08/31/2012')]
        hydrus_recharge = hydrus_recharge[hydrus_recharge.index<pd.to_datetime('09/01/2022')]
        ## Organize the data in blocks
        lookup_dict = {'NE1':['0.5_1','0.5_2','0.5_3'],
                       'NE2':['0.5_6','0.5_7','0.5_8'],
                       'SW1':['0.5_13','0.5_14','0.5_18','0.5_19'],
                       'SW2':['0.5_15','0.5_20'],
                       'NW':['0.5_11','0.5_12','0.5_16','0.5_17'],
                       'SE':['0.5_4','0.5_5','0.5_10','0.5_9'],
                       }
        blockData = pd.DataFrame()
        for block in lookup_dict:
            ## Maybe drop negative values?
            #hydrus_recharge[hydrus_recharge[lookup_dict[block]]<0] = 0
            ## Take the mean across the block and resampling to water year
            blockSub = hydrus_recharge[lookup_dict[block]].mean(axis=1).resample('12MS').sum()
            blockSub.name = block
            blockData = pd.concat([blockData,blockSub],axis=1)
        ## Initialize the figure
        fig, ax = plt.subplots()
        blockData.plot(ax=ax,kind='bar',width=0.8,
                       #colormap='Set2',
                       edgecolor='black',
                       linewidth=0.7,
                       legend=True)
        labelNew = [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for idx,k in enumerate(labels):
            labels[idx] = labelNew[idx]
        ax.set_xticklabels(labels,rotation=60)
        ax.set_title('Modeled Annual Average Recharge')
        ax.set_xlabel('Growing Season')
        ax.set_ylabel('Recharge [cm]')
        print(np.mean((blockData.std(axis=1))))
        print(blockData.mean(axis=1))
        print(np.mean(blockData.mean(axis=1)))
    
    
    def monthly_modeled_recharge(self,resample=False,cumsum=False):
        """
        Plot the modeled annual average recharge per orchard block
        """
        nodeinfo = self.load_data(var='flux')
        nodeinfo = self.clean_data(nodeinfo)
        ## Load the modeled recharge
        hydrus_recharge = self.process_data(quantiles=[0.5],dataset=nodeinfo,loadVar='Flux_299')
        #hydrus_conc = self.process_data(quantiles=[0.5],dataset=nodeinfo,loadVar='Conc_299')
        #hydrus_leaching = 
        mb_recharge = pd.read_csv('/Users/spencerjordan/Documents/Hydrus/python_scripts/massBalance_recharge.csv')
        mb_recharge = mb_recharge.set_index(pd.to_datetime(mb_recharge['Date']))
        mb_recharge = mb_recharge[mb_recharge.index<pd.to_datetime('09/01/2022')]
        ## Slice the data to show only GSs 2013 --> 2022
        hydrus_recharge = hydrus_recharge[hydrus_recharge.index>pd.to_datetime('08/31/2012')]
        hydrus_recharge = hydrus_recharge[hydrus_recharge.index<pd.to_datetime('09/01/2022')]
        ## Resample to a 1-month sum
        hydrus_recharge = hydrus_recharge.mean(axis=1).resample('1m').sum()
        ## Resample mass-balance to 3-month periods and change bar width
        ## I think this looks cleaner than the month-by-month figure
        if resample:
            mb_recharge = mb_recharge.resample('3m').sum()
            hydrus_recharge = hydrus_recharge.resample('3m').sum()
            width=80
        else:
            width = 24
        ## Perform cumsum on both mass balance and hydrus data and changes plot type
        if cumsum:
            mb_recharge = mb_recharge.cumsum()
            hydrus_recharge = hydrus_recharge.cumsum()
            plot_type = 'line'
        else:
            plot_type = 'bar'
        ## Format and plot
        fig, ax = plt.subplots()
        ## Plots a bar graph if plotting normally, plots a line if cumsum is called
        if plot_type == 'bar':
            ax.bar(mb_recharge.index,
                   mb_recharge['avgBalance'],
                   color='blue',
                   width=width,
                   label='Mass Balance Recharge')
        elif plot_type == 'line':
            ax.plot(mb_recharge.index,
                   mb_recharge['avgBalance'],
                   color='blue',
                   label='Mass Balance Recharge')
        ax.plot(hydrus_recharge,
                color='red',
                label='Modeled Recharge')
        ax.set_title('Monthly Recharge - Modeled and Mass Balance')
        ax.set_xlabel('Growing Season')
        ax.set_ylabel('Recharge [cm]')
        ax.legend()
        
        
    def NUE_growth(self,cRoot=0.040):
        ## Load the mass balance to calculate measured NUE_growth
        path = '/Users/spencerjordan/Documents/bowman_data_analysis/N_mass_balance/manual_mass_balance_2022.csv'
        N_balance = pd.read_csv(path)
        N_balance['NUE_growth'] = (N_balance['Uptake kg/ha'] + N_balance['Growth']) / N_balance['Fert kg/ha']
        NUE_measured = N_balance[['block','GS','NUE_growth']]
        ## NE is NaN in GS 2022 because of replanting
        NUE_measured = NUE_measured.fillna(0)
        solute_data = self.clean_data(self.load_data(var='solute'))
        ## Load the applied fertilizer data
        cvTop = self.process_data(quantiles=[0.5],dataset=solute_data,loadVar='Sum(cvTop)').diff().resample('12MS').sum()
        ## Load the root uptake data
        cvRoot = self.process_data(quantiles=[0.5],dataset=solute_data,loadVar='Sum(cvRoot)').diff().resample('12MS').sum()
        start = pd.to_datetime('08/31/2012')
        end = pd.to_datetime('09/01/2022')
        ## Organize data per block and slice to desired timeframe
        cvTop = self.organizeBlocks(cvTop)
        fert = cvTop.set_index(pd.to_datetime(cvTop.index))[(cvTop.index>start)&(cvTop.index<end)]
        cvRoot = self.organizeBlocks(cvRoot)
        treeUptake = cvRoot.set_index(pd.to_datetime(cvRoot.index))[(cvRoot.index>start)&(cvRoot.index<end)]
        NUE_modeled = pd.DataFrame(treeUptake.values / fert.values,
                                    columns=fert.columns,
                                    index=fert.index)
        ## Want pre and post HFLC seperated for plotting
        Modeled_pre = []
        Measured_pre = []
        Modeled_post = []
        Measured_post = []
        ## Compare measured and modeled  using residual sum of squares
        for block in ['NE','SW','SE','NW']:
            measured = NUE_measured['NUE_growth'][NUE_measured['block']==block]
            modeled = NUE_modeled[block]
            ## Drop the tree replanting blocks
            ## Drop the NE block for the replanting year 2022
            ## !!! Also need to dr0p data from other years with immature trees
            if block == 'NE':
                measured = measured[:-1]
                modeled = modeled[:-1]
            ## Pre
            for x in list(modeled.values[:5]):
                Modeled_pre.append(x)
            ## Post
            for x in list(modeled.values[5:]):
                Modeled_post.append(x)
            ## Pre
            for x in list(measured.values[:5]):
                Measured_pre.append(x)
            ## Post
            for x in list(measured.values[5:]):
                Measured_post.append(x)
        ## Plot measured against modeled data
        fig,ax = plt.subplots()
        ax.set_title(f'Model v. Observed Efficiency for growth\ncRoot = {cRoot}',
                     fontsize=11)
        ax.set_ylabel('Modeled')
        ax.set_xlabel('Observed')
        ax.axline((0, 0), slope=1,color='black',ls='dashed')
        ## Plot pre and post data
        ax.scatter(Measured_pre,Modeled_pre,color='red',
                   s=6,
                   label='Pre-HFLC')
        ax.scatter(Measured_post,Modeled_post,color='blue',
                   s=6,
                   label='HFLC')
        ax.set_xlim([0,1.5])
        ax.set_ylim([0,1.5])
        ax.legend()
        plt.show()
    
    
    def nitrate_conc_compare(self):
        """
        Comparison of pore-water N concentrations with nitrate movement in model
        using block average values
        """
        pass
    
    
###############################################################################
################## Interp Hydrus results to MODFLOW model #####################
###############################################################################
class interp_data(plot_hydrus):
    """
    - Try a random selection of concentrations across the field
        - Every 12m is something different/random
        - Assign it a random value from 1 - 20 then pick from there
    - Use 3D geostatistics model to inform where to interpolate the
      data from
    - Run cubic, kriging, NN, average conc/recharge in the area around it?
    
    AgMAR options:
        Interpolated data without any AgMAR --> pass self.no_agmar=True, self.repreat_agmar=False
        Interpolate data with single AgMAR event in May 2022 --> pass self.repeat_agmar=False, self.no_agmar=False
        Interpolate multiple AgMAR events --> pass self.repeat_agmar=True, self.no_agmar=False
            --> Ensure that the dates passed in the 'calc_agamr_x_flux_repeat' functions matches that in the atmosph class
    """
    
    def __init__(self,repeat_agmar=False):
        ## Boolean to control whether or not to repeat AgMAR at some frequency
        self.repeat_agmar = repeat_agmar
        ## Boolean to toggle agmar data interpolation on or off
        self.no_agmar = False
        ## Wells to interpolate from
        self.nwells = [x for x in range(1,21)]
        ## Path to main HYDRUS results
        self.path = '/Users/spencerjordan/Documents/Hydrus/Profiles/result_files/MID_0_92_cRoot_0_040'
        ## Path to HYDRUS AgMAR results
        self.agmar_path = '/Users/spencerjordan/Documents/Hydrus/Profiles/result_files/recharge_cores'
        ## Length of result file
        self.result_length = 51499
        ## Start date of simulation
        self.start_date = pd.to_datetime('09-01-1958',format='%m-%d-%Y')
        ## Pre-define class attributes
        self.grid = None   ## Modflow grid
        self.MW = None     ## Monitoring well GiS locations
        self.orchard = None   ## Orchard boundary
        self.rch = None    ## Recharge basins shapefile
        self.flux_data = None   ## Water flux data
        self.nitrate_data = None   ## Nitrate flux data 
        self.agmar_flux = None   ## Water flux during AgMAR
        self.agmar_nitrate = None  ## Nitrate flux during Agmar
    
    
    def define_hydrus_data(self):
        """
        Load the pickled data using functions from plot_hydrus class
        """
        ## Make sure correct path is loaded, fixes an error 
        self.path = '/Users/spencerjordan/Documents/Hydrus/Profiles/result_files/MID_0_92_cRoot_0_040'
        ## Make sure to load the right data
        self.nwells = [x for x in range(1,21)]
        nodeinfo = self.load_data('flux')
        nodeinfo = self.clean_data(nodeinfo)
        ## Load the Nitrate concentration data
        Conc_299 = self.process_data(dataset=nodeinfo,loadVar='Conc_299',quantiles=[0.5])
        ## Load the water flux data
        Flux_299 = self.process_data(dataset=nodeinfo,loadVar='Flux_299',quantiles=[0.5])
        return Conc_299, Flux_299
    
    
    def create_grid(self,minx_grid,miny_grid,maxx_grid,maxy_grid,N_rows,N_cols):
        """
        Create the spatial grid to represent the top layer of the MODFLOW mesh
        """
        # Grid size
        grid_width = maxx_grid-minx_grid
        grid_height = maxy_grid-miny_grid
        # Cell size
        cell_width = grid_width/N_cols
        cell_height = grid_height/N_rows
        # Define grid origin as upper left grid corner
        origin_y = maxy_grid
        origin_x = minx_grid
        # Create grid cells
        grid_cells = []
        for i in range(N_rows): # For each row
            cell_origin_y = origin_y - i * cell_height # Calculate the current y coordinate
            for j in range(N_cols): # Create all cells in row
                cell_origin_x = origin_x + j * cell_width # Calculate the current x coordinate
                minx_cell = cell_origin_x
                miny_cell = cell_origin_y - cell_height
                maxx_cell = cell_origin_x + cell_width
                maxy_cell = cell_origin_y
                grid_cells.append(box(minx_cell, miny_cell, maxx_cell, maxy_cell)) # Store the new cell
        # Create a GeoDataFrame containing the grid
        self.grid = gpd.GeoDataFrame(geometry=grid_cells)
        
        
    def generate_grid(self,N_ROWS=117,N_COLS=91):
        """
        Generate the grid based on orchard boundary file and model discretization
        """
        ## Bounding box as geopandas dataframe --> Orchard boundary shapefile
        orch = gpd.read_file('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/Model boundary shapefile/Bowman_N_1404x1092.shp')
        orch_bounds = orch.total_bounds
        # Create the grid using the orchard bounding box
        self.create_grid(orch_bounds[0],orch_bounds[1],orch_bounds[2],orch_bounds[3],N_ROWS,N_COLS)


    def define_spatial_data(self):
        """
        Load the spatial data:
        MW locations, orchard boundary, recharge basins, and simulation boundary
        """
        ## Monitoring well coordinates
        mw_coordinates = pd.read_csv('/Users/spencerjordan/Documents/Hydrus/mw_coordinates.csv')
        mw_coordinates['MW'] = np.linspace(1,20,20)
        ## m to ft conversion
        mw_coordinates[['x','y','z']] = mw_coordinates[['x','y','z']]*0.3048
        ## Create spatial data out of mw location using california state plane datum
        crs = "+proj=lcc +lat_1=37.06666666666667 +lat_2=38.43333333333333 +lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000.0000000002 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
        self.MW = gpd.GeoDataFrame(mw_coordinates,
                             geometry=gpd.points_from_xy(x=mw_coordinates['x'],y=mw_coordinates['y'],crs=crs))
        ## Orcahrd boundary
        self.orchard = gpd.read_file('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/Orchard boundary shapefile/Orchard.shp')
        ## Simulation boundary as geopandas object
        self.sim_boundary = gpd.read_file('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/Model boundary shapefile/Bowman_N_1404x1092.shp')
        ## Adding in the recharge basin geometry
        self.rch = gpd.read_file('/Users/spencerjordan/Documents/Hydrus/python_scripts/AgMAR_ShapeFiles/Bowman/Area_gen.shp')
        crs = "+proj=lcc +lat_1=37.06666666666667 +lat_2=38.43333333333333 +lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000.0000000002 +ellps=GRS80 +datum=NAD83 +no_defs"
        self.rch.to_crs(crs=crs,epsg=None,inplace=True)
        ## Create MODFLOW grid, using a function found on SO saved in this directory
        self.generate_grid()
        self.grid['centroid'] = self.grid.centroid
        self.MW['centroid'] = self.MW.centroid
        self.rch['centroid'] = self.rch.centroid
        ## Plot spatial data
        fig,ax = plt.subplots()
        self.sim_boundary.plot(ax=ax,
                               alpha=0.3,
                               color='grey')
        self.orchard.plot(ax=ax,
                          label='orchard boundary')
        self.MW.plot(ax=ax,
                     color='red',
                     markersize=3,
                     label='MWs')
        self.rch.plot(ax=ax,
                      color='orange',
                      label='recharge basin')
        #ax.legend()
        plt.show()


    def plot_spatial_data(self):
        """
        Create plot of the orchard
        """
        base = self.sim_boundary.plot(color='green',
                                      alpha=0.5)
        self.sim_boundary.boundary.plot(ax=base,
                                        color='darkgreen',
                                        label='GW Model Extent')
        self.MW.plot(ax=base,marker='o',color='red',markersize=5,
                label='HYDRUS Profiles\n(Monitoring Wells)')
        #self.rch.boundary.plot(ax=base,label='Recharge Basins',
        #           facecolor='blue')
        self.orchard.boundary.plot(ax=base,color='black',
                              label='Orchard Boundary',
                              lw=1)
        base.legend(fontsize=8,framealpha=1)
        base.set_axis_off()
        base.grid()
        plt.savefig('/users/spencerjordan/Desktop/model_image.png',dpi=300)
    
    
    def calc_water_flux(self,Flux_299,method='nearest'):
        """
        Interpolate the water flux data
        """
        ## Start date of simulations --> 09/01/1987, 30 years shorter than Hydrus sim
        start_day = self.start_date + pd.to_timedelta(32507-(365.25*60),unit='d') 
        Flux_299 = Flux_299[Flux_299.index >= start_day]
        ## Monthly resample of water flux
        flux_monthly = Flux_299.copy().resample('1M').mean()
        ## Spatial grid to interpolate across (model boundary)
        xi = np.vstack((self.grid.centroid.x,self.grid.centroid.y)).T
        ## Points to interpolate from (monitoring well locations)
        points = np.vstack((self.MW.centroid.x,self.MW.centroid.y)).T
        ## Initialize the numpy array with 10647 (91*117) data points for each timestep
        flux_data = np.zeros(10647*len(flux_monthly))
        ## Use the griddata function to interpolate data for each stress-period
        print('Interpolating Water Fluxes')
        for row in range(len(flux_monthly)):
            interped = griddata(points,flux_monthly.iloc[row],xi,method=method)
            ## Fill the NaN values with average if nearest not selected
            if method != 'nearest':
                idx = np.isnan(interped)
                avgConc = np.mean(interped[~idx])
                ## Set NaN values to values from the nearest interpolation
                interped[idx] = avgConc

            flux_data[row*10647:(row+1)*10647] = interped
        self.flux_data = flux_data
            
        
    def calc_nitrate_flux(self,Conc_299,method='nearest'):
        """
        Interpolate the nitrate flux data
        """
        ## Start date of simulations
        start_day = self.start_date + pd.to_timedelta(32507-(365.25*60),unit='d') 
        Conc_299 = Conc_299[Conc_299.index >= start_day]
        ## Monthly resample of for water flux
        conc_monthly = Conc_299.copy().resample('1M').mean()
        xi = np.vstack((self.grid.centroid.x,self.grid.centroid.y)).T
        points = np.vstack((self.MW.centroid.x,self.MW.centroid.y)).T
        conc_data = np.zeros(10647*len(conc_monthly))
        ## Use the griddata function to interpolate data
        print('Interpolating Nitrate Concentrations')
        for row in range(len(conc_monthly)):
            interped = griddata(points,conc_monthly.iloc[row],xi,method=method)
            ## If using cubic or linear interpolation, need to fill the NaN values with nearest method
            if method != 'nearest':
                idx = np.isnan(interped)
                avgConc = np.mean(interped[~idx])
                interped[idx] = avgConc
            conc_data[row*10647:(row+1)*10647] = interped
        self.nitrate_data = conc_data
        
        
    def define_agmar_data(self):
        """
        Load and define the AgMAR data, currently using predicted recharge from core
        8 only. 
        """
        ########################################
        ## Load the data from each recharge well
        ########################################
        ## Change path to the recharge simulations
        self.path = self.agmar_path
        ## Change nwells to only recharge data --> using as a proxy for all of the recharge basins
        self.nwells = [6,7,8]
        ## Load and process AgMAR data
        nodeinfo = self.load_data('flux')
        nodeinfo = self.clean_data(nodeinfo)
        ## Load the Nitrate concentration data
        nitrate_agmar = self.process_data(dataset=nodeinfo,loadVar='Conc_299',quantiles=[0.5])
        ## Process the agmar Flux data
        flux_agmar = self.process_data(dataset=nodeinfo,loadVar='Flux_299',quantiles=[0.5])
        ## Set the CRS of grid to be the same as the recharge basin shapefile
        crs = "+proj=lcc +lat_1=37.06666666666667 +lat_2=38.43333333333333 +lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000.00000000    02 +ellps=GRS80 +datum=NAD83 +no_defs"
        self.grid.set_crs(crs=crs,epsg=None,inplace=True)
        ## Find the intersection of the recharge basins and modflow grid
        ## Basin above well 6
        basin_6 = gpd.sjoin(self.grid, self.rch.loc[self.rch['Comment']=='BMO-P1'])
        ## Basin above well 7
        basin_7 = gpd.sjoin(self.grid, self.rch.loc[self.rch['Comment']=='BMO- P2'])
        ## Basin above well 8
        basin_8 = gpd.sjoin(self.grid, self.rch.loc[self.rch['Comment']=='BMO-P3'])
        ## Basins as a list
        rch_basins = [basin_6,basin_7,basin_8]
        self.flux_agmar = flux_agmar
        self.nitrate_agmar = nitrate_agmar
        self.rch_basins = rch_basins
    
    
    def calc_agmar_water_flux(self,Flux_299):
        """
        Create subset of data for AgMAR flooding water fluxes
        """
        print('Calculating AgMAR Fluxes')
        ## Start and end of AgMAR
        ## !!! Hard coded to only include the actual AgMAR that occured at the orcahrd
        start_day = self.start_date + pd.to_timedelta(32507-(365.25*60)+12662,unit='d') 
        end_day = start_day + pd.to_timedelta(58,unit='d')
        ## Grab the subset of original flux values during recharge period
        ## NOT resampling to monthly here since AgMAR is under daily stress-periods in MODFLOW
        original_flux = Flux_299.copy()
        original_flux = original_flux[original_flux.index >= start_day] 
        original_flux = original_flux[original_flux.index <= end_day]
        ## Same interpolation as with regular data
        xi = np.vstack((self.grid.centroid.x,self.grid.centroid.y)).T
        points = np.vstack((self.MW.centroid.x,self.MW.centroid.y)).T
        flux_data = np.zeros(10647*len(original_flux))
        
        ########################################################
        ## Interpolate data during AgMAR --> daily stress period
        ########################################################
        for row in range(len(original_flux)):
            interped = griddata(points,original_flux.iloc[row],xi,method='nearest')
            ## Add the data for each of the three recharge basins
            for idx,well in enumerate([6,7,8]):
                interped[self.rch_basins[idx].index.values] = self.flux_agmar[f'0.5_{well}'][row+23254]
            flux_data[row*10647:(row+1)*10647] = interped
        self.agmar_flux = flux_data
        
        #############################################################################
        ## Interpolate basin data to basin area after AgMAR --> monthly stress period
        #############################################################################
        if self.no_agmar:
            pass
        else:
            flux_dat = self.flux_data.copy()
            agmar_data = self.flux_agmar.copy()
            agmar_data = agmar_data[agmar_data.index>end_day].resample('1M').mean()
            for row in range(len(agmar_data)):
                ## !!! Index starts after AgMAR
                dat_index = row + 417
                ## Subset to edit with recharge basin data
                flux_to_edit = flux_dat[dat_index*10647:(dat_index+1)*10647]
                ## Loop through for each of the monitoring wells being flooded
                for idx,well in enumerate([6,7,8]):
                    ## Replace existing data with data from the AgMAR simulation
                    flux_to_edit[self.rch_basins[idx].index.values] = agmar_data[f'0.5_{well}'][row]
                ## Assign the edited data back into the original dataset
                flux_dat[dat_index*10647:(dat_index+1)*10647] = flux_to_edit
            self.flux_data = flux_dat

        
    def calc_agmar_nitrate_flux(self,Conc_299):
        """
        Create subset of data for AgMAR flooding nitrate fluxes 
        """
        print('Calculating AgMAR Nitrate')
        ## Start and end of AgMAR
        start_day = self.start_date + pd.to_timedelta(32507-(365.25*60)+12662,unit='d') 
        end_day = start_day + pd.to_timedelta(58,unit='d')
        ## Grab the subset of original flux values during recharge period
        ## NOT resampling to monthly here since AgMAR is under daily stress-periods in MODFLOW
        original_conc = Conc_299.copy()
        original_conc = original_conc[original_conc.index >= start_day] 
        original_conc = original_conc[original_conc.index <= end_day]
        ## Same interpolation as with regular data
        xi = np.vstack((self.grid.centroid.x,self.grid.centroid.y)).T
        points = np.vstack((self.MW.centroid.x,self.MW.centroid.y)).T
        nitrate_data = np.zeros(10647*len(original_conc))
        
        ###################################################
        ## Interpolate basin data to basin area after AgMAR
        ###################################################
        for row in range(len(original_conc)):
            interped = griddata(points,original_conc.iloc[row],xi,method='nearest')
            for idx,well in enumerate([6,7,8]):
                interped[self.rch_basins[idx].index.values] = self.nitrate_agmar[f'0.5_{well}'][row+23254]
            nitrate_data[row*10647:(row+1)*10647] = interped
        self.agmar_nitrate = nitrate_data
        
        ###################################################
        ## Interpolate basin data to basin area after AgMAR
        ###################################################
        if self.no_agmar:
            pass
        else:
            N_dat = self.nitrate_data
            agmar_data = self.nitrate_agmar.copy()
            agmar_data = agmar_data[agmar_data.index>end_day].resample('1M').mean()
            for row in range(len(agmar_data)):
                ## !!! Index starts after AgMAR
                dat_index = row + 417
                ## Subset to edit with recharge basin data
                N_to_edit = N_dat[dat_index*10647:(dat_index+1)*10647]
                ## Loop through for each of the monitoring wells being flooded
                for idx,well in enumerate([6,7,8]):
                    ## Replace existing data with data from the AgMAR simulation
                    N_to_edit[self.rch_basins[idx].index.values] = agmar_data[f'0.5_{well}'][row]
                ## Assign the edited data back into the original dataset
                N_dat[dat_index*10647:(dat_index+1)*10647] = N_to_edit
            self.nitrate_data = N_dat
        
        
    ###########################################################################
    ################## Interpolate multiple AgMAR events ######################
    ###########################################################################
    def calc_agmar_water_flux_repeat(self,Flux_299):
        """
        Create subset of data for AgMAR flooding water fluxes
        """    
        print('Calculating Repeating AgMAR Fluxes')
        ## List of start days to repeat --> MUST match the startDays list in
        ## the atmosph class when input files were made
        startDays = [pd.to_datetime('05/02/2022',format='%m/%d/%Y')]
        ## Repeat flooding every 10th Winter on January 1st
        for i in range(7):
            startDays.append(pd.to_datetime(f'01/01/20{32+i*10}',format='%m/%d/%Y'))
        agmarFluxDict = {}
        for k,start_day in enumerate(startDays):
            ## Start and end of AgMAR
            end_day = start_day + pd.to_timedelta(58,unit='d')
            ## Grab the subset of original flux values during recharge period
            ## NOT resampling to monthly here since AgMAR is under daily stress-periods in MODFLOW
            original_flux = Flux_299.copy()
            ## Find row index for flooding
            agmar_row_idx = original_flux.reset_index().loc[original_flux.reset_index()['day']==start_day].index[0]
            original_flux = original_flux[original_flux.index >= start_day] 
            original_flux = original_flux[original_flux.index <= end_day]
            ## Same interpolation as with regular data
            xi = np.vstack((self.grid.centroid.x,self.grid.centroid.y)).T
            points = np.vstack((self.MW.centroid.x,self.MW.centroid.y)).T
            flux_data = np.zeros(10647*len(original_flux))
            ########################################################
            ## Interpolate data during AgMAR --> daily stress period
            ########################################################
            for row in range(len(original_flux)):
                interped = griddata(points,original_flux.iloc[row],xi,method='nearest')
                ## Add the data for each of the three recharge basins
                for idx,well in enumerate([6,7,8]):
                    interped[self.rch_basins[idx].index.values] = self.flux_agmar[f'0.5_{well}'][row+agmar_row_idx]
                flux_data[row*10647:(row+1)*10647] = interped
            agmarFluxDict[k] = flux_data
        ## Assign to class variable to that it can be passed to pickle function
        self.agmarFluxDict = agmarFluxDict
        
        #############################################################################
        ## Interpolate basin data to basin area after AgMAR --> monthly stress period
        #############################################################################
        flux_dat = self.flux_data.copy()
        agmar_data = self.flux_agmar.copy()
        agmar_data = agmar_data[agmar_data.index>pd.to_datetime('2022-06-29 00:00:00')].resample('1M').mean()
        for row in range(len(agmar_data)):
            ## !!! Index starts after AgMAR
            dat_index = row + 417
            ## Subset to edit with recharge basin data
            flux_to_edit = flux_dat[dat_index*10647:(dat_index+1)*10647]
            ## Loop through for each of the monitoring wells being flooded
            for idx,well in enumerate([6,7,8]):
                ## Replace existing data with data from the AgMAR simulation
                flux_to_edit[self.rch_basins[idx].index.values] = agmar_data[f'0.5_{well}'][row]
            ## Assign the edited data back into the original dataset
            flux_dat[dat_index*10647:(dat_index+1)*10647] = flux_to_edit
        self.flux_data = flux_dat
        
        
    def calc_agmar_nitrate_flux_repeat(self,Conc_299):
        """
        Create subset of data for AgMAR flooding nitrate fluxes 
        """
        print('Calculating Repeating AgMAR Nitrate')
        ## List of start days to repeat --> MUST match the startDays list in
        ## the atmosph class when input files were made
        startDays = [pd.to_datetime('05/02/2022',format='%m/%d/%Y')]
        ## Repeat flooding every 10th Winter on January 1st
        for i in range(7):
            startDays.append(pd.to_datetime(f'01/01/20{32+i*10}',format='%m/%d/%Y'))
        agmarNitrateDict = {}
        for k,start_day in enumerate(startDays):
            end_day = start_day + pd.to_timedelta(58,unit='d')
            ## Grab the subset of original flux values during recharge period
            ## NOT resampling to monthly here since AgMAR is under daily stress-periods in MODFLOW
            original_conc = Conc_299.copy()
            agmar_row_idx = original_conc.reset_index().loc[original_conc.reset_index()['day']==start_day].index[0]
            original_conc = original_conc[original_conc.index >= start_day] 
            original_conc = original_conc[original_conc.index <= end_day]
            ## Same interpolation as with regular data
            xi = np.vstack((self.grid.centroid.x,self.grid.centroid.y)).T
            points = np.vstack((self.MW.centroid.x,self.MW.centroid.y)).T
            nitrate_data = np.zeros(10647*len(original_conc))
            ######################################################
            ## Interpolate basin data to basin area after AgMAR
            ######################################################
            for row in range(len(original_conc)):
                interped = griddata(points,original_conc.iloc[row],xi,method='nearest')
                for idx,well in enumerate([6,7,8]):
                    interped[self.rch_basins[idx].index.values] = self.nitrate_agmar[f'0.5_{well}'][row+agmar_row_idx]
                nitrate_data[row*10647:(row+1)*10647] = interped
            agmarNitrateDict[k] = nitrate_data
        ## Assign to class variable to that it can be passed to pickle function
        self.agmarNitrateDict = agmarNitrateDict
        
        ######################################################
        ## Interpolate basin data to basin area after AgMAR
        ######################################################
        N_dat = self.nitrate_data
        agmar_data = self.nitrate_agmar.copy()
        agmar_data = agmar_data[agmar_data.index>pd.to_datetime('2022-06-29 00:00:00')].resample('1M').mean()
        for row in range(len(agmar_data)):
            ## !!! Index starts after AgMAR
            dat_index = row + 417
            ## Subset to edit with recharge basin data
            N_to_edit = N_dat[dat_index*10647:(dat_index+1)*10647]
            ## Loop through for each of the monitoring wells being flooded
            for idx,well in enumerate([6,7,8]):
                ## Replace existing data with data from the AgMAR simulation
                N_to_edit[self.rch_basins[idx].index.values] = agmar_data[f'0.5_{well}'][row]
            ## Assign the edited data back into the original dataset
            N_dat[dat_index*10647:(dat_index+1)*10647] = N_to_edit
        self.nitrate_data = N_dat
 
    
    def create_outputs(self):
        ## Initialize dataframes        
        ## MODFLOW stress periods
        stressPeriods = list(range(1,1345))
        stressPeriods_array = np.repeat(stressPeriods, [10647]*1344)
        ## For recharge
        stressPeriods_agmar = list(range(1,60))
        stressPeriods_agmar = np.repeat(stressPeriods_agmar, [10647]*59)
        ## Columns in the modflow grid
        cols = list(range(1,118))
        cols = np.repeat(cols, [91]*117).tolist()
        ## For AgMAR and normal sim
        cols_agmar = cols * 59
        cols_array= cols * 1344
        ## Rows in the MODFLOW model grid
        rows = list(range(1,92))*117
        ## Water flux data for MODFLOW
        flux = pd.DataFrame(columns=['row','col','recharge','sp'])
        rows_agmar = rows * 59
        rows_array = rows * 1344
        flux['sp'] = stressPeriods_array
        flux['row'] = rows_array
        flux['col'] = cols_array
        ## Nitrate concentrations --> no data yet so can copy everything
        nitrate = flux.copy()
        ## AgMAR flux data
        flux_agmar = pd.DataFrame(columns=['row','col','recharge','sp'])
        flux_agmar['sp'] = stressPeriods_agmar
        flux_agmar['row'] = rows_agmar
        flux_agmar['col'] = cols_agmar
        ## AgMAR nitrate data
        nitrate_agmar = flux_agmar.copy()
        #######################################
        ########### Input the Data ############
        #######################################
        ## If doing multi-agmar
        if self.repeat_agmar:
            flux_agmarDict = {}
            nitrate_agmarDict = {}
            ## Process each flooding event and enter into a dict
            for i in self.agmarFluxDict.keys():
                flux_agmar['recharge'] = self.agmarFluxDict[i]
                nitrate_agmar['C'] = self.agmarNitrateDict[i]
                flux_agmarDict[i] = flux_agmar
                nitrate_agmarDict[i] = nitrate_agmar
            ## Only need these once
            flux['recharge'] = self.flux_data
            nitrate['C'] = self.nitrate_data
            ## Return the dict data
            return flux, nitrate, flux_agmarDict, nitrate_agmarDict
        ## No multi-agmar
        else:
            ## Process data normally
            flux['recharge'] = self.flux_data
            nitrate['C'] = self.nitrate_data
            flux_agmar['recharge'] = self.agmar_flux
            nitrate_agmar['C'] = self.agmar_nitrate
            ## Return the single agmar event
            return flux, nitrate, flux_agmar, nitrate_agmar
        
        
    def pickle_outputs(self):
        """
        Pickle the output files to be easily read in MODFLOW Flopy script
        """
        path = '/Users/spencerjordan/Documents/Hydrus/python_scripts/agmar_pickle_files'
        ## Path to output pickle files to be read by flopy
        if self.repeat_agmar:
            ## Now have a dict containing the data from each AgMAR flooding
            flux, nitrate, flux_agmarDict, nitrate_agmarDict = self.create_outputs()
            ## Only need to save these once
            pickle.dump(flux, open(f'{path}/recharge.p', "wb" ))
            pickle.dump(nitrate, open(f'{path}/nitrate.p', "wb" ))
            for i in self.agmarFluxDict.keys():
                flux_agmar = flux_agmarDict[i]
                nitrate_agmar = nitrate_agmarDict[i]
                pickle.dump(flux_agmar, open(f'{path}/recharge_agmar_{i}.p', "wb" ))
                pickle.dump(nitrate_agmar, open(f'{path}/nitrate_agmar_{i}.p', "wb" ))
        else:
            ## Load and pickle all the data, only one agmar event
            flux, nitrate, flux_agmar, nitrate_agmar = self.create_outputs()
            pickle.dump(flux, open(f'{path}/recharge.p', "wb" ))
            pickle.dump(nitrate, open(f'{path}/nitrate.p', "wb" ))
            pickle.dump(flux_agmar, open(f'{path}/recharge_agmar.p', "wb" ))
            pickle.dump(nitrate_agmar, open(f'{path}/nitrate_agmar.p', "wb" ))


    def plot_contours(self):
        """
        Plot the first timestep for each of the interpolated timeseries
        Mainly use this as a check to make sure it interpolated correctly
        """
        plt.imshow(self.flux_data[0:10647].reshape(117,91))
        plt.title('Water Flux')
        plt.show()
        plt.imshow(self.nitrate_data[0:10647].reshape(117,91))
        plt.title('Nitrate Conc')
        plt.show()
        plt.imshow(self.agmar_flux[10647:21294].reshape(117,91))
        plt.title('AgMAR Water Flux')
        plt.show()
        plt.imshow(self.agmar_nitrate[10647:21294].reshape(117,91))
        plt.title('AgMAR Nitrate Conc')
        plt.show()
    
    
    def plot_avg_recharge(self):
        with open('/Users/spencerjordan/Documents/Hydrus/python_scripts/agmar_pickle_files/recharge.p','rb') as handle:
            recharge_df = pickle.load(handle)
        recharge_df = recharge_df.groupby('sp').mean()
        recharge_df['recharge'].plot()
        plt.ylabel('Water Flux [cm/day]')
        plt.title('Water Flux per Stress Period')
    
    
    def interpolate(self):
        """
        Run all relevant functions and pickle the resulting numpy arrays
        Defaults to running AgMAR, but can pass False if not needed
        """
        ## Load water flux and nitrate data
        Conc_299,Flux_299 = self.define_hydrus_data()
        self.flux = Flux_299
        ## Define the spatial data, such as the orchard boundary and spatial grid
        self.define_spatial_data()
        ## Load the water flux and nitrate data predicted by AgMAR simulations
        self.define_agmar_data()
        ## Interpolate the water flux data across all stress periods
        self.calc_water_flux(Flux_299,method='nearest')
        ## Interpolate the nitrate concentrations across all stress periods
        self.calc_nitrate_flux(Conc_299,method='nearest')
        ## If using repeated AgMAR sims, ask user to input the frequency
        if self.repeat_agmar:
            self.calc_agmar_water_flux_repeat(Flux_299)
            self.calc_agmar_nitrate_flux_repeat(Conc_299)
            ## Pickle the outputs to be read by modflow script
            self.pickle_outputs()
        ## Else, interpolate water/nitrate flux for AgMAR simulations for single flooding
        else:
            self.calc_agmar_water_flux(Flux_299)
            self.calc_agmar_nitrate_flux(Conc_299)
            ## Pickle the outputs to be read by modflow script
            self.pickle_outputs()

    
        
        
        
        
    
