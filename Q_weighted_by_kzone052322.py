#Find proportion layers are contributing to Q in well

import flopy
import numpy as np
import os
from matplotlib import pyplot as plt
#import shapefile 
import pandas as pd
import geopandas as gpd
import itertools 
import random
import pickle
import matplotlib.dates as mdates

#%% Pickle function definition
# Set up optimization: python to accompany UCODE

# Load mf once through flopy then use pickle
def object_save(thing=object, name=str):      
    file_to_store = open(name+'.pickle', 'wb')
    pickle.dump(thing, file_to_store)
    file_to_store.close()
    
def object_load(name=str):
    file_to_read = open(name+'.pickle', "rb")
    thing = pickle.load(file_to_read)
    file_to_read.close()
    return thing

PATH = '/Users/spencerjordan/Documents/modflow_work'


#%%

# #Initialize empty recarray
all_particle_data = pd.DataFrame()
btc = pd.DataFrame({'Date':[],'MW':[],'N':[], 'run':[]})

# #APPEND
#all_particle_data = pd.concat((all_particle_data, mp_result), ignore_index=True)

## Changing to what I ran, only did the three i had tprog files for - Spencer

## List of tsim realizations modeled in Model_runs... script
tsim_runs = [1]


for run in tsim_runs:
    try: 
        mf = object_load(f'{PATH}/mf')
    except: 
        pass

    print(run)
    
    p = gpd.read_file('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/MW location shapefile/mw_meters.shp')
    # Set up grid intersection with model grid
    # dz=0.32m; wells screened from water table (7m) to 14m; layers 18-50 
    
    ## ASSUMES YOU ALREADY HAVE MODFLOW LOADED AS MF
    ix = flopy.utils.gridintersect.GridIntersect(mf.modelgrid, method='vertex')
    mw_node = {} #dictionary of nodes corresponding to cells for each well; MP7 needs nodes not layer,row,column for get_destination_pathline_data to work
    for i in range(0,len(p.geometry)):
        locs = list() #one list per well
        for j in range(21,43): #layers
            locs.append((j,) + ix.intersects(p.geometry[i])[0][0]) #need list of tuples; add layers of interest to tuple
        mw_node[i] = mf.dis.get_node(locs)
    
    mw_node_list = list(itertools.chain(*mw_node.values()))
    
    lpf = mf.get_package('LPF')
    #Or recreate tsim from lpf
    tsim = np.stack(lpf.hk.get_value())
    reference = np.sort(np.unique(tsim))[::-1] #sort from largest to smallest
    for i in range(0,4):
        tsim[tsim==reference[i]] = i+1
    
    tsim_file = '/Users/spencerjordan/Documents/modflow_work/Geology/tsim_BowmanN_1404x1092.asc'+str(run)
    tsim = np.loadtxt(tsim_file,dtype='int', skiprows=1)
    tsim = tsim.reshape((125,117,91)) #nlay, nrow,ncol
    tsim = np.flip(tsim,axis=[0,1]) #origin of tsim is bottom corner (-20m); layer 125, need to flip upside down
    tsim = abs(tsim)
    
    # tsim = np.loadtxt('C:/Users/rache/Desktop/GWV simple model/GWVplay/BowmanN/1404x1092/tsim_bowmanN_1404x1092.dat',dtype='int', delimiter=' ')
    # tsim = tsim.reshape((nlay,nrow,ncol))
   
    #params = pd.read_csv('params.txt',header=None)[0:4].to_numpy()[:,0]
    #k1,k2,k3,k4 = params[0:4]
    k1,k2,k3,k4 = reference
    
    zone_tracking = {'well':list(),'kzone':list()}
    for i in range(0,20):
        nodes = mw_node.get(i) #nodes for well
        lrc = mf.dis.get_lrc(nodes)
        for j in range(len(lrc)):
            zone_tracking['well'].append(i+1)
            zone_tracking['kzone'].append(tsim[lrc[j]])
        
    zone_data_well = pd.DataFrame(zone_tracking)
    zone_data_well['layer'] = np.tile(np.arange(21,43),20)
    zone_data_well.to_csv('zone_data_well.csv')
    
    zone_tracking={'well':list(), 'zone1':list(),'zone2':list(),'zone3':list(),'zone4':list()}
    for i in range(1,21):
        subbywell = zone_data_well[zone_data_well['well'] == i] #get data by well
        zone_tracking['well'].append(i)
        zone_tracking['zone1'].append(len(subbywell[subbywell.kzone==1])/len(subbywell))
        zone_tracking['zone2'].append(len(subbywell[subbywell.kzone==2])/len(subbywell))
        zone_tracking['zone3'].append(len(subbywell[subbywell.kzone==3])/len(subbywell))
        zone_tracking['zone4'].append(len(subbywell[subbywell.kzone==4])/len(subbywell))
    zone_data_well_frac = pd.DataFrame(zone_tracking)
        
    #Sove system of equations for say Q=1
    Q_frac = {'well':list(), 'zone1':list(),'zone2':list(),'zone3':list(),'zone4':list()}
    Q = 1
    for i in range(0,20):
        b = [Q,0,0,0]
        A1 = zone_data_well_frac.zone1[i]
        A2 = zone_data_well_frac.zone2[i]
        A3 = zone_data_well_frac.zone3[i]
        A4 = zone_data_well_frac.zone4[i]
    
        A = [[1,1,1,1],[-1/(k1*A1), 1/(k2*A2), 0, 0], [0, -1/(k2*A2), 1/(k3*A3), 0], [0,0,-1/(k3*A3), 1/(k4*A4) ]]
        
        if A3==0:
            b = [Q, 0,0]
            A = [[1,1,1],[-1/(k1*A1), 1/(k2*A2), 0], [0, -1/(k2*A2), 1/(k4*A4)]]
            Q_frac['well'].append(i+1)
            Q_frac['zone1'].append(np.linalg.solve(A, b)[0])
            Q_frac['zone2'].append(np.linalg.solve(A, b)[1])
            Q_frac['zone3'].append(0)
            Q_frac['zone4'].append(np.linalg.solve(A, b)[2])
            
        else:
            Q_frac['well'].append(i+1)
            Q_frac['zone1'].append(np.linalg.solve(A, b)[0])
            Q_frac['zone2'].append(np.linalg.solve(A, b)[1])
            Q_frac['zone3'].append(np.linalg.solve(A, b)[2])
            Q_frac['zone4'].append(np.linalg.solve(A, b)[3])
    
    Q_frac = pd.DataFrame(Q_frac)
    
    if (Q_frac.isna().to_numpy().flatten().sum() >0): 
        print('WARNING: NA on run ' + str(run+1))
    
    zone_data_well['Q_frac'] = -999
    for i in range(1,21):
        for j in range(1,5): #kzones
            zone_data_well.loc[(zone_data_well['well']==i) & (zone_data_well['kzone'] == j),'Q_frac'] = Q_frac.iloc[i-1,j]/(zone_data_well_frac.iloc[i-1,j]*22)
    
    #################### MT3D Breakthrough Curve Plots#############################
    fname = '/Users/spencerjordan/Documents/modflow_work/recharge_model/MT3D001.UCN'
    #fname = '/Volumes/Spencer Drive/modflow_work/bowman_no_agmar/MT3D001.UCN'
    ucnobj = flopy.utils.UcnFile(fname)
    times = ucnobj.get_times()
    time = pd.to_timedelta(times,'D')+np.datetime64('1987-09-01')
    

    conc = ucnobj.get_alldata()
    conc[conc==1e30]=0
    
    # Get well locations from shapefile
    p=gpd.read_file('/Users/spencerjordan/Documents/modflow_work/GW_modflow_Model-selected/Input data/MW location shapefile/mw_meters.shp')
    # Set up grid intersection with model grid
    ix = flopy.utils.gridintersect.GridIntersect(mf.modelgrid, method='vertex')
    
    
    ########################################################
    ################## LOAD MT3D DATA HERE #################
    ########################################################
    btc = pd.DataFrame({'Date':[],'MW':[],'N':[], 'run':[]})
    for i in range(0,len(p.geometry)):
        print(i)
        #sub = zone_data_well[(zone_data_well['well']==(i+1)) & (zone_data_well['kzone'] <3)] #do this when you only want zone 1 and 2
        sub = zone_data_well[zone_data_well['well']==(i+1)]
        row,col = ix.intersects(p.geometry[i])[0][0]
        extract_conc = conc[:,sub['layer'],row,col]
        
        #print(extract_conc.min()*1000, extract_conc.max()*1000)
        for j in range(0,len(extract_conc)):
            mw_conc = (extract_conc[j]*sub.Q_frac).sum()*1000 #mg/L 
            #mw_conc = (extract_conc[j].mean())*1000 #mg/L do this when you want to weight them all equally
            df = pd.DataFrame({'Date':[time[j]], 'MW':[i+1], 'N': [mw_conc], 'run':[run+1]})
            #btc = btc.append(df, ignore_index=True)
            btc = pd.concat([btc,df],ignore_index=True)
    '''
        
    # #Get data  from modpath
    # # # Can read in endpoint data but I don't think it's very accurate
    # fpth = os.path.join(os.getcwd(), mf.name+'_mp.mpend') #filepath for endpoint data
    # #fpth = 'C:\\Users\\rache\\Desktop\\Flopy\\Modflow input\\Bowman1404x1092_ss_mp.mpend'
    # e = flopy.utils.EndpointFile(fpth)
    # ew = e.get_alldata()
    # #ew = e.get_destination_endpoint_data(mw_node_list, source=True) #get endpoints for MW1 points
    
    # #ew[ew['status']==7]#These are the ones that terminate inside the model area
    
    # #e.write_shapefile(endpoint_data = ew, shpname='endpoints.shp', direction='ending', sr=mf.sr)
    # #e.write_shapefile(endpoint_data = ew[ew['status']==7], shpname='endpoints.shp', direction='ending', sr=mf.sr)
    
    
    # # Use endpoints to find travel distance and travel time
    # # One line per particle: need time0 and time
    # #Only use particles that terminated
    # #ew = ew[(ew['status']==7)]
    # e_time = (ew['time']-ew['time0'])/365.25 #Travel time in years for each point in well
    # e_dist = np.sqrt((ew['x']-ew['x0'])**2+(ew['y']-ew['y0'])**2)
    # #If xloc or yloc is 0 then it left the model area
    
    # tracking = {'particleid':list(), 'particlegroup':list(), 'traveltime':list(), 'distance':list(), 
    #             'x_i':list(), 'y_i':list(), 'z_i':list(),
    #             'x_f':list(),'y_f':list(),'z_f':list()}
                        
    # for i in np.unique(ew['particleid']): #for all particles
    #     # pw_subset = pw[pw['particlegroup']==0] #Gets data just for one particle group (one well); multiple particles
    #     ew_subset = ew[ew['particleid']==i] #Gets data for one particle
    #     time = (ew_subset['time']-ew_subset['time0'])/365.25 
    #     dist = np.sqrt((ew_subset['x']-ew_subset['x0'])**2+(ew_subset['y']-ew_subset['y0'])**2)
    
    #     tracking['particleid'].append(i)
    #     tracking['particlegroup'].append(ew_subset['particlegroup'][0])
    #     tracking['x_i'].append(ew_subset.x0[0])
    #     tracking['y_i'].append(ew_subset.y0[0])
    #     tracking['z_i'].append(ew_subset.z0[0])
    #     tracking['x_f'].append(ew_subset.x[0])
    #     tracking['y_f'].append(ew_subset.y[0])
    #     tracking['z_f'].append(ew_subset.z[0])
        
    #     if ((ew_subset['status']==5)): # made it through recharge
    #         print(time,dist)
    #         tracking['traveltime'].append(time[0])
    #         tracking['distance'].append(dist[0])
    
    #     else:
    #         tracking['traveltime'].append(-999)
    #         tracking['distance'].append(-999)
    
    
    # data = pd.DataFrame(tracking)
    
    #Pathline
    fpth = os.path.join(f'{PATH}/hanni_modpath/Results_Run_tsim{run}/{mf.name}_mp.mppth') #filepath for mppth data
    #fpth = 'C:\\Users\\rache\\Desktop\\Flopy\\Modflow input\\Bowman1404x1092_ss_mp.mppth'
    p = flopy.utils.PathlineFile(fpth)
    #p.write_shapefile(pw, one_per_particle=False, shpname='pathlines.shp',sr=mf.sr)
    pw = p.get_destination_pathline_data(mw_node_list, to_recarray=True) #get pathlines for MW screen points
    
    h = flopy.utils.HeadFile(f'{PATH}/hanni_modpath/Results_Run_tsim{run}/{mf.name}.hds', model= mf)
    #h = flopy.utils.HeadFile('C:\\Users\\rache\\Desktop\\Flopy\\Modflow input\\Bowman1404x1092_ss.hds', model= mf)
    hds = h.get_alldata()
    wt = flopy.utils.postprocessing.get_water_table(hds, nodata=mf.hdry)  #water table
    
    tracking = {'particleid':list(), 'particlegroup':list(), 'traveltime':list(), 'distance':list(), 
                'x_i':list(), 'y_i':list(), 'z_i':list(),
                'x_f':list(),'y_f':list(),'z_f':list()}
                        
    for i in np.unique(pw['particleid']): #for all particles
        #print(i)
        pw_subset = pw[pw['particleid']==i] #Gets data for one particle
        pw_start = pw_subset[0] #particle start
        #pw_end = pw_subset[-1]
        # try: 
        #     pw_end = pw_subset[np.where(pw_subset['k']==(np.min(pw_subset['k'])+1))[0][0]] #Use secondlayer from where it gets stuck   (old way)
        # except:
        #     pw_end = pw_subset[np.where(pw_subset['k']==(np.min(pw_subset['k'])+0))[0][0]]
        
        min_loc = np.ones(len(pw_subset))*-999
        for j in range(len(pw_subset)):
            node = pw_subset.node[j]
            lrc = mf.dis.get_lrc(node)
            min_loc[j] = abs(wt[pw_subset.stressperiod[j], lrc[0][1], lrc[0][2]] - pw_subset.z[j])
            #min_loc[j] = abs(wt[lrc[0][1], lrc[0][2]] - pw_subset.z[j])
        
        try:
            pw_end = pw_subset[np.min(np.where(min_loc<0.1))] #had done min_loc.argmin but it actually bounces; find first
        
            time = (pw_end['time']-pw_start['time'])/365.25
            dist = np.sqrt((pw_end['x']-pw_start['x'])**2+(pw_end['y']-pw_start['y'])**2) #This is surface distance not total distance traveled
        except:
            time = -999
            dist = -999
            
        print(time)
        tracking['particleid'].append(i)
        tracking['particlegroup'].append(pw_subset['particlegroup'][0])
        tracking['x_i'].append(pw_start.x)
        tracking['y_i'].append(pw_start.y)
        tracking['z_i'].append(pw_start.z)
        tracking['x_f'].append(pw_end.x)
        tracking['y_f'].append(pw_end.y)
        tracking['z_f'].append(pw_end.z)
        
        # if (min_loc.min() <=0.32):
        #     print(min_loc.min())
        tracking['traveltime'].append(time)
        tracking['distance'].append(dist)
   
        # else:
        #     tracking['traveltime'].append(-999)
        #     tracking['distance'].append(-999)
        
        
        # if ((ew['status']==7)[i]==True):
        #     tracking['traveltime'].append(time)
        #     tracking['distance'].append(dist)
        
        # if ((ew['status']==7)[i]==False):
        #     tracking['traveltime'].append(-999)
        #     tracking['distance'].append(-999)
        #     print('oops!')
    
    data = pd.DataFrame(tracking)
    
    #zone_data_sub = zone_data_well[zone_data_well.kzone < 3]
    zone_data_sub = zone_data_well
    
    #Apply weighting by Q fraction at that position in well screen. Only had particles where kzone is 1 or 2, and then 10 particles per cell.
    data = pd.concat((data,np.repeat(zone_data_sub.kzone,10).reset_index(drop=True)),axis=1)
    data = pd.concat((data,np.repeat(zone_data_sub.Q_frac,10).reset_index(drop=True)/10),axis=1)
    
    #data = data[data['kzone']<3] #limit to just zones 1 and 2
    data = data[data['traveltime']>0]
    Q_frac['zone1']+Q_frac['zone2'] 
    
    data['run'] = run+1
    data.to_csv('/Users/spencerjordan/Documents/modflow_work/particle_runs/particle_data_homo_allzone052322.csv', header=False, index=False, mode='a') #save particle ending points
    
    mp_result = pd.DataFrame({'MW':[],'time_w':[],'dist_w':[],'flow_fraction':[]})
    for i in data.particlegroup.unique(): #by well
        sub = data[data.particlegroup==i]
        surfaced = sub[sub.traveltime>0]
        plt.hist(surfaced.loc[surfaced.kzone<=2,'traveltime'],range=(0,60))
        plt.xlabel('Years')
        plt.ylabel('# particles')
        plt.title('GW Age: MW '+str(i+1))
        plt.show()
        
        plt.hist(surfaced.loc[surfaced.kzone<=2,'distance'])
        plt.xlabel('Distance (meters)')
        plt.ylabel('# particles')
        plt.title('Source area distance: MW '+str(i+1))
        plt.show()
        
        time_w = sum(surfaced.traveltime*surfaced.Q_frac)
        dist_w = sum(surfaced.distance*surfaced.Q_frac)
        fraction = surfaced.Q_frac.sum()
        df = pd.DataFrame({'MW':[i+1],'time_w':[time_w],'dist_w':[dist_w],'flow_fraction':[fraction]})
        mp_result = mp_result.append(df,ignore_index=True)
    
    all_particle_data = pd.concat((all_particle_data, mp_result), ignore_index=True)
    
all_particle_data.to_csv(f'{PATH}/mp_result051922.csv')


#%%Plot all the results
#load_data = pd.read_csv('C:/Users/rache/Desktop/Flopy/Modflow input/Runs040722/particle_data.csv')
#load_data = pd.read_csv('D:/Models/Flopy/Runs041322/Test params w same tsim/particle_data_params.csv')
#load_data = pd.read_csv('D:/Models/Flopy/Runs041322/Different tsim same param/Rest of it/particle_data_tsim.csv')

## Changed this file name to something that is almost certainly the wrong file --> Spencer
load_data = pd.read_csv('{PATH}/mp_result051922.csv')
load_data_homo = pd.read_csv('/Users/spencerjordan/Documents/modflow_work/particle_runs/particle_data_homo_allzone052322.csv')
## Added this because I did not see the regular particle data file...
load_data = load_data_homo
load_data.columns =['particleid', 'particlegroup', 'traveltime', 'distance', 'x_i', 'y_i', 'z_i', 'x_f', 'y_f', 'z_f', 'kzone', 'Q_frac', 'run']#,'x','y']

#Histograms of particles
for i in load_data.particlegroup.unique(): #by well
    sub = load_data[load_data.particlegroup==i]
    surfaced = sub[sub.traveltime>0]
    plt.hist(surfaced.loc[surfaced.kzone<=2,'traveltime'],range=(0,60))
    plt.xlabel('Years')
    plt.ylabel('# particles')
    plt.title('GW Age: MW '+str(i+1))
    #plt.savefig('GW_age_MW'+str(i+1))
    plt.show()
    
    plt.hist(surfaced.loc[surfaced.kzone<=2,'distance'])
    plt.xlabel('Distance (meters)')
    plt.ylabel('# particles')
    plt.title('Source area distance: MW '+str(i+1))
    #plt.savefig('Source_area_MW'+str(i+1))
    plt.show()
 
#Flow-weighted PDFs
for i in load_data.particlegroup.unique(): #by well
    sub = load_data[load_data.particlegroup==i]
    surfaced = sub[sub.traveltime>0]
    plt.hist(surfaced.loc[surfaced.kzone<=2,'traveltime'],range=(0,60),weights=surfaced.loc[surfaced.kzone<=2,'Q_frac'],density=True)
    plt.xlabel('Years')
    plt.ylabel('PDF')
    plt.title('GW Age: MW '+str(i+1))
    #plt.savefig('GW_age_MW'+str(i+1))
    plt.show()
    
    plt.hist(surfaced.loc[surfaced.kzone<=2,'distance'], weights=surfaced.loc[surfaced.kzone<=2,'Q_frac'], density=True)
    plt.xlabel('Distance (meters)')
    plt.ylabel('PDF')
    plt.title('Source area distance: MW '+str(i+1))
    #plt.savefig('Source_area_MW'+str(i+1))
    plt.show()   
    
#Plot all wells on one
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,8),gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1,1]})
sub1 = load_data_params
sub2 = load_data_tsim
sub3 = load_data_homo
surfaced3 = sub3[sub3.traveltime>0]
surfaced1 = sub1[sub1.traveltime>0]
onerun1 = surfaced1[surfaced1.run==1]
surfaced2 = surfaced2[surfaced2.traveltime>0]
onerun2 = surfaced2[surfaced2.run==1]

#Plot all runs
ax1.hist(surfaced1.loc[surfaced1.kzone<=2,'traveltime'], alpha=0.8,
          bins=60,range=(0,60),
          weights=surfaced1.loc[surfaced1.kzone<=2,'Q_frac'],
          density=True)
#Plot one run on top to show variability in one run
ax1.hist(onerun1.loc[onerun1.kzone<=2,'traveltime'], alpha=0.5,
          bins=60,range=(0,60),
          weights=onerun1.loc[onerun1.kzone<=2,'Q_frac'],
          density=True)

#Plot homogeneous model
ax1.hist(surfaced3.loc[surfaced3.kzone<=2,'traveltime'], alpha=0.5,
          bins=60,range=(0,60),
          weights=surfaced3.loc[surfaced3.kzone<=2,'Q_frac'],
          density=True)

ax1.legend(['all runs','one run'],loc='upper left')
ax1.set_ylim([0,0.8])
#ax1.set_xlabel('Years')
ax1.set_ylabel('PDF')
ax1.set_title('GW age',size=12)

ax3.hist(surfaced2.loc[surfaced2.kzone<=2,'traveltime'], alpha=0.8,
         bins=60,range=(0,60),
         weights=surfaced2.loc[surfaced2.kzone<=2,'Q_frac'],
         density=True)
ax3.hist(onerun2.loc[onerun2.kzone<=2,'traveltime'], alpha=0.5,
          bins=60,range=(0,60),
          weights=onerun2.loc[onerun2.kzone<=2,'Q_frac'],
          density=True)
ax3.hist(surfaced3.loc[surfaced3.kzone<=2,'traveltime'], alpha=0.5,
          bins=60,range=(0,60),
          weights=surfaced3.loc[surfaced3.kzone<=2,'Q_frac'],
          density=True)



ax3.legend(['all runs','one run'],loc='upper left')
ax3.set_ylim([0,0.8])
ax3.set_xlabel('Years')
ax3.set_ylabel('PDF')
#ax3.set_title('GW age',size=12)

ax2.hist(surfaced1.loc[surfaced1.kzone<=2,'distance'], alpha=0.8,
         bins=100, 
         weights=surfaced1.loc[surfaced1.kzone<=2,'Q_frac'], 
         density=True)
ax2.hist(onerun1.loc[onerun1.kzone<=2,'distance'], alpha=0.5,
          bins=100,
          weights=onerun1.loc[onerun1.kzone<=2,'Q_frac'],
          density=True)
ax2.hist(surfaced3.loc[surfaced3.kzone<=2,'distance'], alpha=0.5,
          bins=100,
          weights=surfaced3.loc[surfaced3.kzone<=2,'Q_frac'],
          density=True)


ax2.legend(['all runs','one run'])
ax2.set_xlim([0,1000])
ax2.set_ylim([0,0.0045])
#ax2.set_xlabel('Distance (meters)')
#ax2.set_ylabel('PDF')
ax2.set_title('Source area distance',size=12)

ax4.hist(surfaced2.loc[surfaced2.kzone<=2,'distance'], alpha=0.8,
         bins=100, 
         weights=surfaced2.loc[surfaced2.kzone<=2,'Q_frac'], 
         density=True)
ax4.hist(onerun2.loc[onerun2.kzone<=2,'distance'], alpha=0.5,
          bins=100,
          weights=onerun2.loc[onerun2.kzone<=2,'Q_frac'],
          density=True)
ax4.hist(surfaced3.loc[surfaced3.kzone<=2,'distance'], alpha=0.5,
          bins=100,
          weights=surfaced3.loc[surfaced3.kzone<=2,'Q_frac'],
          density=True)

ax4.legend(['all runs','one run'])
ax4.set_xlim([0,1000])
ax4.set_ylim([0,0.0045])
ax4.set_xlabel('Distance (meters)')
#ax4.set_ylabel('PDF')
#ax4.set_title('Source area distance',size=12)

plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.85)
plt.figtext(0.5,0.95, "A. Varying hydraulic parameters, \n constant heterogeneous geology", ha="center", va="top", fontsize=14, color="k")
plt.figtext(0.5,0.5, "B. Varying heterogeneous geology simulation, \n constant hydraulic parameter set", ha="center", va="top", fontsize=14, color="k")
fig.suptitle('Particle Tracking Results for All 20 MWs', y=1.01, fontsize=16)

######### Plot particle starting depth vs. travel time and distance
fig, ((ax1,ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(10,8),gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
sub1 = load_data_params
sub2 = load_data_tsim
ax1.scatter(sub1.loc[sub1.kzone==2,'traveltime'], sub1.loc[sub1.kzone==2,'z_i'] - 3.75, s=1, color='k', alpha=0.5)
ax1.scatter(sub1.loc[sub1.kzone==1,'traveltime'], sub1.loc[sub1.kzone==1,'z_i']- 3.75, s=1, color='orange', alpha=0.5)

ax1.scatter(sub2.loc[sub2.kzone==2,'traveltime'], sub2.loc[sub2.kzone==2,'z_i']- 3.75, s=1, color='k', alpha=0.5)
ax1.scatter(sub2.loc[sub2.kzone==1,'traveltime'], sub2.loc[sub2.kzone==1,'z_i']- 3.75, s=1, color='orange', alpha=0.5)

ax2.scatter(sub1.loc[sub1.kzone==2,'distance'], sub1.loc[sub1.kzone==2,'z_i']- 3.75, s=1, color='k', alpha=0.5)
ax2.scatter(sub1.loc[sub1.kzone==1,'distance'], sub1.loc[sub1.kzone==1,'z_i']- 3.75, s=1, color='orange', alpha=0.5)

ax2.scatter(sub2.loc[sub2.kzone==2,'distance'], sub2.loc[sub2.kzone==2,'z_i']- 3.75, s=1, color='k', alpha=0.5)
ax2.scatter(sub2.loc[sub2.kzone==1,'distance'], sub2.loc[sub2.kzone==1,'z_i']- 3.75, s=1, color='orange', alpha=0.5)

ax2.legend(['f.sand','c.sand'], title='hydrofacies \n intersecting \nwell screen')
ax1.set_xlabel('GW age (years)')
ax1.set_ylabel('Approximate height above bottom of well screen (m)')
ax2.set_xlabel('Source area distance (m)')
fig.suptitle('Particle backtracking result \n relation to starting height in well screen', y=0.95, fontsize=16)


#Check for stabilization of CV
CV_overall=pd.DataFrame()
for i in range(0,1):
    CV_save=pd.DataFrame()
    random.seed=i
    for k in range(26,27):
        run_set = random.sample(range(1,27),k)
        data = load_data[load_data.run.isin(run_set)]
        
        mp_result = pd.DataFrame({'MW':[],'time_w':[],'dist_w':[],'flow_fraction':[]})
        for j in run_set:
            for i in data.particlegroup.unique(): #by well
                sub = data[(data.particlegroup==i) & (data.run==j)]
                surfaced = sub[sub.traveltime>0]
        
                time_w = sum(surfaced.traveltime*surfaced.Q_frac)
                dist_w = sum(surfaced.distance*surfaced.Q_frac)
                fraction = surfaced.Q_frac.sum()
                df = pd.DataFrame({'MW':[i+1],'time_w':[time_w],'dist_w':[dist_w],'flow_fraction':[fraction]})
                mp_result = mp_result.append(df,ignore_index=True)
                
        std_mp = mp_result.groupby(by='MW').std()
        mean_mp = mp_result.groupby(by='MW').mean()
        #CV_save = CV_save.append(((std_mp/mean_mp).mean(axis=0)), ignore_index=True)
        CV_save = CV_save.append(mean_mp, ignore_index=True)

        CV_overall = CV_overall.append({'nruns':k,'time_CV':CV_save.time_w.mean(),'dist_CV':CV_save.dist_w.mean()}, ignore_index=True)

fig, ax = plt.subplots()

ax.fill_between(CV_overall.nruns.unique(), 
                CV_overall.groupby(by='nruns').min().time_CV.to_numpy(),  
                CV_overall.groupby(by='nruns').max().time_CV.to_numpy(),
                alpha=0.5)

ax.fill_between(CV_overall.nruns.unique(), 
                CV_overall.groupby(by='nruns').min().dist_CV.to_numpy(),  
                CV_overall.groupby(by='nruns').max().dist_CV.to_numpy(),
                alpha=0.5)
ax.plot(CV_overall.nruns.unique(), 
                CV_overall.groupby(by='nruns').mean().time_CV.to_numpy())
ax.plot(CV_overall.nruns.unique(), 
                CV_overall.groupby(by='nruns').mean().dist_CV.to_numpy())
    
ax.set_xlabel('Number of parameter sets')
ax.set_ylabel('CV')
plt.suptitle('Model Stabilization under Parameter Uncertainty', y=1.05, fontsize=12)
plt.title('Average CV across 20 wells \n +/- range over randomly selected sets', fontsize=10)
plt.legend(['average age','average distance','age range','distance range'])
plt.show()

data['time_w'] = float('NaN')#set it up to ignore (nan) points that have traveltime >60 yrs
data.loc[data.traveltime>0,'time_w']= data[data.traveltime>0]['traveltime'] * data[data.traveltime>0]['Q_frac'] 
data['dist_w'] = float('NaN')#set it up to ignore (nan) points that have traveltime >60 yrs
data.loc[data.traveltime>0,'dist_w']= data[data.traveltime>0]['distance'] * data[data.traveltime>0]['Q_frac']

weighted_time = (data.groupby(['particlegroup']).sum()/len(data.run.unique())).time_w.to_numpy()
weighted_dist = (data.groupby(['particlegroup']).sum()/len(data.run.unique())).dist_w.to_numpy()

plt.hist(weighted_time, bins=np.arange(0,60,5))
plt.title('Representative GW Age in 20 MWs')
plt.xlabel('Years')
plt.show()

plt.hist(weighted_dist)
plt.title('Representative GW Source Area Distance in 20 MWs')
plt.xlabel('Meters')
plt.show()

#Try finding weighted time and distance only using coarse and fine sand zones (1 and 2)
#Also reporting what fraction of the well volume that represents

sub = data[data['kzone']<3] #limit to just zones 1 and 2
weighted_time = sub.groupby(['particlegroup']).sum().time_w.to_numpy()
weighted_dist = sub.groupby(['particlegroup']).sum().dist_w.to_numpy()

Q_frac['zone1']+Q_frac['zone2'] #This proportion of the flow is represented (>90% for all wells but MW9 is ~87%)

plt.hist(weighted_time, bins=np.arange(0,100,10))
plt.title('Representative GW Age in 20 MWs')
plt.xlabel('Years')
plt.show()

plt.hist(weighted_dist)
plt.title('Representative GW Source Area Distance in 20 MWs')
plt.xlabel('Meters')
plt.show()

plt.bar(np.arange(1,21), height=weighted_time/weighted_time.max(), width=0.3)
plt.bar(np.arange(1,21)+0.3, height=weighted_dist/weighted_dist.max(), width=0.3)
#plt.bar(np.arange(1,21)+0.6, height=(zone_data_well_frac['zone1']+zone_data_well_frac['zone2']), width=0.3)
plt.xticks(np.arange(1,21))
plt.xlabel('well')
plt.ylabel('Normalized age or distance, coarse fraction')
plt.title('Normalized MW GW age and source area distance')
plt.legend(['GW age','Distance','% coarse'])
plt.show()


#Very similar to thoe other plot

#Save mp results
#mean (unweighted)
#traveltime = data.groupby(['particlegroup']).mean().traveltime.to_numpy()
#distance = data.groupby(['particlegroup']).mean().distance.to_numpy()
#mp_results = list()
#mp_results.append(np.concatenate((param_val,weighted_time,weighted_dist, traveltime,distance)))
        

# Get data from mt3d
#From measured nitrate data (look @ Reinterp_mw_nitrate.py)
sample_date = hds['Sampling Date'].unique()
#Find closest model transport step
transport_step = list()
for day in sample_date:
    transport_step.append(abs(time-day).argmin())
    
#Go by well 
    
# Get well locations from shapefile
p = gpd.read_file(f'{PATH}/MW location shapefile/mw_meters.shp')
# Set up grid intersection with model grid
ix = flopy.utils.gridintersect.GridIntersect(mf.modelgrid, method='vertex')

#btc = pd.DataFrame({'Date':[],'MW':[],'N':[]})
for i in range(0,len(p.geometry)):
    print(i)
    sub = zone_data_well[zone_data_well['well']==(i+1)]
    row,col = ix.intersects(p.geometry[i])[0][0]
    extract_conc = conc[:, sub['layer'],row,col]
    print(extract_conc.min()*1000, extract_conc.max()*1000)
    for j in range(0,len(extract_conc)):
        mw_conc = (extract_conc[j]*sub.Q_frac).sum()*1000 #mg/L
        df = pd.DataFrame({'Date':[time[j]], 'MW':[i+1], 'N': [mw_conc]})
        btc = btc.append(df, ignore_index=True)

## Plots entire N timeseries across the wells
for i in range(1,21):
    btc[btc.MW==i].plot(x='Date',y='N')
    #plt.plot(hds[hds['MW']==('MW'+str(i))]['Sampling Date'], hds[hds['MW']==('MW'+str(i))][['NO3-N (mg/L)']])
    plt.title('Breakthrough curve: MW '+str(i))
    plt.show()        
        
# #Apply weighting by Q fraction at that position in well screen
# zone_data_well['conc_w']=zone_data_well.conc*zone_data_well.Q_frac
# weighted_conc = zone_data_well.groupby(['well']).sum().conc_w.to_numpy()*1000 #mg/L
# measured = hds.loc[(hds.index.year == 2020) & (hds.index.month == 8)] #august is closest

# plt.bar(np.arange(1,21), height=measured['NO3-N (mg/L)'], width=0.4)
# plt.bar(np.arange(1,21)+0.4, height=weighted_conc, width=0.4)
# plt.xticks(np.arange(1,21))
# plt.xlabel('well')
# plt.ylabel('Nitrate (mg/L)')
# plt.title('MW nitrate concentration, July 2020')
# plt.legend(['Measured','Model'])
# plt.show()

## Save or read in current files
#btc.to_csv('homo_btc.csv')
#btc = pd.read_csv('params_tsim1_btc.csv')
btc.Date = pd.to_datetime(btc.Date)
'''

#%% Just to get rid of btc_rch error --> Only for analysis between rch or original sims
#btc_rch = object_load(f'{PATH}/btc_rch')

## Use this if you loaded recharge data with script
#btc = object_load(f'{PATH}/btc_rch')
#btc = object_load(f'{PATH}/btc')

#btc_rch = btc
##############################################
## ALL OF THESE ARE FOR WELLS 6, 7, AND 8 ONLY
## AgMAR scenario files
##############################################
btc_avg_15 = object_load(f'{PATH}/python_scripts/btc_15')
btc_avg_40 = object_load(f'{PATH}/python_scripts/btc_40')
btc_avg_60 = object_load(f'{PATH}/python_scripts/btc_60')
btc_avg_100 = object_load(f'{PATH}/python_scripts/btc_100')
## AgMAR every 10-years
btc_x10 = object_load(f'{PATH}/python_scripts/btc_x10')
## No AgMAR, reference
btc_ref = object_load(f'{PATH}/python_scripts/btc_ref')

#%% Load analytical Data
mw_coordinates = pd.read_csv('/Users/spencerjordan/Documents/Hydrus/mw_coordinates.csv')
mw_coordinates['MW#'] = np.arange(1,21)

#Get WL depth measurements (ft)
hds = pd.read_csv('/Users/spencerjordan/Documents/bowman_data_analysis/BOW-MW-ALL-DATA-Compiled_working.csv')
#Get rid of nans
hds = hds[hds['NO3-N (mg/L)'].notna()]
hds = hds.merge(mw_coordinates, on='MW#', how='left')

hds['Sampling Date'] = pd.to_datetime(hds['Sampling Date'])
hds = hds.set_index('Sampling Date')


#%%
#fig, ax = plt.subplots()
#for i in range(1,21):
for i in [6,7,8]:
    plt.plot(hds[hds['MW']==('MW'+str(i))].index, hds[hds['MW']==('MW'+str(i))][['NO3-N (mg/L)']],label='Measured')#,marker='o',ms=2.0)
    
    for j in [2]:
        ## Load the btc variable twice and rename it after - bad bad coding
        #sub_rch = btc_rch[(btc_rch.MW==i) & (btc_rch.run==j)]
        #plt.plot(sub_rch.Date, sub_rch.N,label='Simulated - Recharge',color='r')
        ## Also hard indexing j here=
        sub = btc[(btc.MW==i) & (btc.run==j)]
        plt.plot(sub.Date, sub.N,label='Simulated - Original',color='purple')
        #plt.ylim([0,80])
        plt.xlabel('Year')
        plt.ylabel('N concentration (mg/L)')
        plt.title('MW'+str(i)+' Breakthrough Curve')
    #plt.axvline(x=pd.to_datetime('2018'),ls='--',color='black',label='Switch to HFLC')
    y = sub['N'][sub.Date==pd.to_datetime('2022/05/02')]
    #plt.plot(pd.to_datetime('2022/05/02'),y,marker='*',ms=12,color='green',
    #         label='Recahrge Event')
    plt.title(f'MW{i}')
    #plt.xlim([pd.to_datetime('2015',format='%Y'),pd.to_datetime('2030',format='%Y')])
    #plt.ylim([0,4])
    plt.legend()
    #plt.savefig(f'/Users/spencerjordan/Documents/AGU_figures_2022/modflow_compar{i}.png',dpi=150)
    plt.grid()
    plt.show()
    
#%% Recreate everage N for all MW's
#btc = btc_ref
"""
btc_6 = btc[btc['MW']==6]  
btc_7 = btc[btc['MW']==7]  
btc_8 = btc[btc['MW']==8]    
"""

hds_6 = hds[hds['MW#']==6]  
hds_7 = hds[hds['MW#']==7]  
hds_8 = hds[hds['MW#']==8]  

hds_new = pd.concat([hds_6,hds_7,hds_8])
#btc_new = pd.concat([btc_6,btc_7,btc_8])

hds_avg = hds_new.groupby(by=hds_new.index).mean()    
hds_05 = hds_new.groupby(by=hds_new.index).min()
hds_95 = hds_new.groupby(by=hds_new.index).max()

#btc_avg = btc_new.groupby(by=['Date']).mean()    
#btc_05 = btc_new.groupby(by=['Date']).quantile([0.05])        
#btc_95 = btc_new.groupby(by=['Date']).quantile([0.95])

#btc_rch_avg = btc_rch_new.groupby(by=['Date']).mean()    
#btc_rch_05 = btc_rch_new.groupby(by=['Date']).max()    
#btc_rch_95 = btc_rch_new.groupby(by=['Date']).min() 


"""
###############################################
## FOR ORIGINAL PLOT WITH WITH DEFAULT LINES ##
###############################################
hds_avg = hds.groupby(by=hds.index).mean(numeric_only=True)    
hds_min = hds.groupby(by=hds.index).min()
hds_max = hds.groupby(by=hds.index).max()
#hds_min = hds.groupby(by=hds.index).quantile([0.05])    
#hds_max = hds.groupby(by=hds.index).quantile([0.95])    

btc_avg = btc.groupby(by=['Date']).mean()    
btc_05 = btc.groupby(by=['Date']).quantile([0.05])    
btc_95 = btc.groupby(by=['Date']).quantile([0.95])
btc_05 = btc.groupby(by=['Date']).min()
btc_95 = btc.groupby(by=['Date']).max()
###############################################
"""
## Initialize the plot
fig, ax = plt.subplots(figsize=(12,8))
plt.xticks(rotation=45)

#ax.axvline(x=pd.to_datetime(2018,format='%Y'),ls='--',color='black',
#           label='Swtich to HFLC')
#ax.axvline(x=pd.to_datetime('2022-05-03',format='%Y-%m-%d'),ls='--',color='blue',
#           label='May 2022 AgMAR',
#           alpha=0.6)
#ax.axhline(10,ls='--',color='red',
#           label='10 mg/L MCL')

#ax.axvline(x=pd.to_datetime('2022/05/02'),ls='--',color='blue',
#           label='AgMAR Event')


## for making AgMAR comparison plots --> Load these via pickle file
ax.plot(btc_avg_15.index,btc_avg_15.N,label='15% applied water')
ax.plot(btc_avg_40.index,btc_avg_40.N,label='40% applied water')
ax.plot(btc_avg_60.index,btc_avg_60.N,label='60% applied water')
ax.plot(btc_avg_100.index,btc_avg_100.N,label='100% applied water',
        color='mediumorchid')
ax.plot(btc_x10.index,
        btc_x10.N,
        label='100% applied water\nEvery 10 years',
        color='mediumorchid')

ax.plot(btc_ref.index,btc_ref.N,label='No AgMAR')

#ax.fill_between(btc_avg.index,btc_05.N,btc_95.N,alpha=0.5,
#                color='#8CBE92')
## Plot field data
"""
ax.plot(hds_avg.index,
        hds_avg['NO3-N (mg/L)'],
        label='Field Data',
        ls='dashdot',
        lw=2,
        color='darkblue')
"""


"""
ax.fill_between(hds_avg.index,hds_05['NO3-N (mg/L)'],hds_95['NO3-N (mg/L)'],alpha=0.3,
                color='purple')
"""

'''
ax.annotate('~30 year delay in modeled nitrate decrease\nfrom start of HFLC',(.45,.6),
            xycoords='figure fraction',
            fontsize=14,
            )
'''

#ax.annotate('v',(pd.to_datetime('2047-03',format='%Y-%m'),48),
#            fontsize=14,
#            )
#ax.vlines(pd.to_datetime(2048,format='%Y'),48,70,color='black')

"""
ax.plot(btc_rch_avg.index,btc_rch_avg.N, color='orange',
        label='AgMAR Simulation')
   
ax.fill_between(btc_rch_avg.index,btc_rch_05.N,btc_rch_95.N,
                alpha=0.5,color='orange')
"""

ax.legend(fontsize=15)
ax.set_title('Modeled Average NO3-N Concentrations at Wells 6,7,8',
             fontsize=20)
#ax.annotate('Mean Concentration and Spatial Variability Across Single Realization',
#            (0.08,0.9),
#            xycoords='figure fraction',
#            fontsize=17)

ax.tick_params(labelsize=16)
ax.set_ylabel('NO3-N (mg/L)',fontsize=20)
ax.set_xlabel('Year',fontsize=20)
ax.xaxis.set_major_locator(mdates.YearLocator((5)))

ax.set_xlim([pd.to_datetime(2005,format='%Y'),pd.to_datetime(2100,format='%Y')])    

#plt.savefig('/Users/spencerjordan/Documents/CWEMF_figures/longTermNitrateCompare_origSim.png',dpi=200,
#            bbox_inches='tight')


## Creating a weighted average



#%% Figure for GRA WGC
x = [['A panel', 'B panel',]]
gs_kw = dict(width_ratios=[1, 2])
fig,ax = plt.subplot_mosaic(x,
                            figsize=(10, 5), 
                            layout="constrained",
                            gridspec_kw=gs_kw,
                            facecolor='#D9E8FC')

fig.suptitle('Average NO3-N Concentrations at Wells 6,7,8 Following AgMAR:\nMeasured and Simulated',
             fontsize=16)
fig.supylabel('NO3-N (mg/L)',fontsize=16)
fig.supxlabel('Year',fontsize=16)

ax['B panel'].plot(btc_avg_15.index,
        btc_avg_15.N,
        label='15% applied water')
ax['B panel'].plot(btc_avg_40.index,
        btc_avg_40.N,
        label='40% applied water')
#ax['B panel'].plot(btc_avg_60.index,
#        btc_avg_60.N,
#        label='60% applied water')
#ax['B panel'].plot(btc_avg_100.index,
#        btc_avg_100.N,
#        label='100% applied water')
ax['B panel'].plot(btc_x10.index,
        btc_x10.N,
        label='100% applied water\nEvery 10 years')
ax['B panel'].plot(btc_ref.index,
        btc_ref.N,
        label='No AgMAR')

ax['B panel'].axvspan(xmin=pd.to_datetime('05/03/2022'),
                      xmax=pd.to_datetime('05/31/2120'),
                      alpha=0.25,
                      color='grey',
                      label='Post AgMAR')

ax['B panel'].grid()
ax['B panel'].set_xlim([pd.to_datetime(2005,format='%Y'),
                        pd.to_datetime(2100,format='%Y')])    
ax['B panel'].legend(fontsize=10)


##################
## Left side panel
##################

ax['A panel'].axvspan(xmin=pd.to_datetime('05/03/2022'),
                      xmax=pd.to_datetime('05/31/2028'),
                      alpha=0.3,
                      color='grey',
                      label='Post AgMAR')

ax['A panel'].plot(hds_avg.index,
        hds_avg['NO3-N (mg/L)'],
        label='GW Well Data',
        lw=2,
        color='darkblue')
ax['A panel'].plot(btc_avg_100.index,
        btc_avg_100.N,
        label='Simulated',
        color='darkred')
ax['A panel'].set_xlim([pd.to_datetime(2015,format='%Y'),
                        pd.to_datetime(2026,format='%Y')])    
ax['A panel'].grid()
ax['A panel'].legend()

plt.savefig('/users/spencerjordan/Desktop/agmar_wgc_fig.png',
            dpi=300)

#%%

for i in [x+1 for x in range(20)]:
    head = hds.loc[hds['MW#']==i,:]
    fig,ax = plt.subplots()
    head['NO3-N (mg/L)'].plot(ax=ax)
    
    mt3d = btc.loc[btc['MW']==i,:]
    mt3d.set_index('Date',inplace=True)
    mt3d['N'].plot(ax=ax)

    ax.set_title(f'Monitoring Well {i}')
    ax.grid()
    #ax.set_xlim([pd.to_datetime(2015,format='%Y'),pd.to_datetime(2030,format='%Y')])    


















