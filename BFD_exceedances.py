#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# # BFD_exceedances.ipynb

# In[1]:


'''
    File name: BFD_exceedances.ipynb
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 04.07.2024
    Date last modified: 04.07.2024

    ############################################################## 
    Purpose:

    Read in WBGT data from CONUS404 control and PGW that were processed in:
    projects/2020_CONUS404/programs/CONUS404_preprocessor/GWBT_code/PyWBGT-1.0.0/Jupyter_notebooks/Calculate_WBGT_with_C404_data.py

    Calculate monthly exceedances of flag warnings for each hour of the day

    Save data to netcdf files for further processing
    
'''


# In[2]:


from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
# import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import scipy
import shapefile
import matplotlib.path as mplPath
from matplotlib.patches import Polygon as Polygon2
# Cluster specific modules
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.vq import kmeans2,vq, whiten
from scipy.ndimage import gaussian_filter
import seaborn as sns
# import metpy.calc as mpcalc
import shapefile as shp
import sys
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import pickle
import time
import xarray as xr
from tqdm import tqdm
import matplotlib.gridspec as gridspec


# In[3]:


years = [int(sys.argv[1])]

# datasets = ['ERA5', 'CONUS404_ctr', 'CONUS404_pgw']
datasets = ['CONUS404_pgw']

flag_thresholds = [29, 31, 32]

savedir = '/glade/campaign/mmm/c3we/prein/Papers/2024/2024_WBGT-climate-change/Flag_Exceedances/'


# ### loop over datasets and do the processing

# In[9]:


for da in range(len(datasets)):
    print('WORK ON '+datasets[da])

    if datasets[da] == 'ERA5':
        data_dir = '/glade/campaign/mmm/c3we/ESTCP/ERA5/WBGT/'
        time = pd.date_range(datetime.datetime(1950, 1, 1, 0), 
                              end=datetime.datetime(2019, 12, 31, 23), freq='h')
        
        conus_era5 = [100,1250,270,900]
        
        # read ERA5 coordinates
        ncid=Dataset('/glade/campaign/mmm/c3we/ESTCP/ERA5/WBGT/WBGT_ERA5_2008.nc', mode='r')
        lon = np.array(np.squeeze(ncid.variables['longitude'][conus_era5[3]:conus_era5[1]]))
        lat = np.array(np.squeeze(ncid.variables['latitude'][conus_era5[0]:conus_era5[2]]))
        ncid.close()

        # years = np.unique(time.year)
        wbgt = np.zeros((len(years), 12, 24, len(flag_thresholds), len(lat), len(lon))); wbgt[:] = np.nan
        for yy in tqdm(range(len(years))):
            outfile = savedir + str(years[yy]) + '_' + datasets[da] + '_flag_day_frequency.npz'
            if os.path.isfile(outfile) == False:
            
                file = data_dir + 'WBGT_ERA5_'+str(years[yy])+'.nc'
                ncid=Dataset(file, mode='r')
                data_tmp = np.array(np.squeeze(ncid.variables['WBGT'][:,conus_era5[0]:conus_era5[2],conus_era5[3]:conus_era5[1]]))
                ncid.close()
    
                for mm in range(12):
                    data_mon = data_tmp[time[time.year == years[yy]].month == (mm+1)]
                    data_mon = np.reshape(data_mon, (int(data_mon.shape[0]/24), 24, len(lat), len(lon)))
                    for th in range(len(flag_thresholds)):
                        wbgt[mm,:,th,:,:] = np.sum(data_mon-273.15 >= flag_thresholds[th], axis=0)
    
                # save the data
                np.savez(outfile,
                        lon = lon,
                        lat = lat,
                        flag_thresholds = flag_thresholds,
                        exceedances = wbgt[:,:,:,:,:])
            
    elif datasets[da] == 'CONUS404_ctr':
        data_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData/GWBT/'
        time = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                              end=datetime.datetime(2022, 12, 31, 23), freq='h')
        
        # read CONUS404 coordinates
        ncid=Dataset('/glade/campaign/ncar/USGS_Water/CONUS404/wrfconstants_d01_1979-10-01_00:00:00.nc4', mode='r')
        lon = np.array(np.squeeze(ncid.variables['XLONG'][:]))
        lat = np.array(np.squeeze(ncid.variables['XLAT'][:]))
        ncid.close()

        # years = np.unique(time.year)
        for yy in range(len(years)):
            wbgt = np.zeros((12, 24, len(flag_thresholds), lat.shape[0], lat.shape[1])); wbgt[:] = np.nan
            outfile = savedir + str(years[yy]) + '_' + datasets[da] + '_flag_day_frequency.npz'
            if os.path.isfile(outfile) == False:
                for mm in tqdm(range(12)):
                    file = data_dir + 'GWBT_'+str(years[yy])+str(mm+1).zfill(2)+'_CONUS404.nc'
                    ncid=Dataset(file, mode='r')
                    data_tmp = np.array(np.squeeze(ncid.variables['GWBT'][:]))
                    ncid.close()
                    data_mon = np.reshape(data_tmp, (int(data_tmp.shape[0]/24), 24, lat.shape[0], lat.shape[1]))                    
                    for th in range(len(flag_thresholds)):
                        wbgt[mm,:,th,:,:] = np.sum(data_mon-273.15 >= flag_thresholds[th], axis=0)
    
                # save the data
                np.savez(outfile,
                        lon = lon,
                        lat = lat,
                        flag_thresholds = flag_thresholds,
                        exceedances = wbgt[:,:,:,:,:])
        
    elif datasets[da] == 'CONUS404_pgw':
        data_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData_PGW/GWBT/'
        time = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                              end=datetime.datetime(2022, 12, 31, 23), freq='h')
        
        # read CONUS404 coordinates
        ncid=Dataset('/glade/campaign/ncar/USGS_Water/CONUS404/wrfconstants_d01_1979-10-01_00:00:00.nc4', mode='r')
        lon = np.array(np.squeeze(ncid.variables['XLONG'][:]))
        lat = np.array(np.squeeze(ncid.variables['XLAT'][:]))
        ncid.close()

        # years = np.unique(time.year)
        for yy in range(len(years)):
            wbgt = np.zeros((12, 24, len(flag_thresholds), lat.shape[0], lat.shape[1])); wbgt[:] = np.nan
            outfile = savedir + str(years[yy]) + '_' + datasets[da] + '_flag_day_frequency.npz'
            if os.path.isfile(outfile) == False:
                for mm in tqdm(range(12)):
                    file = data_dir + 'GWBT_'+str(years[yy])+str(mm+1).zfill(2)+'_CONUS404.nc'
                    ncid=Dataset(file, mode='r')
                    data_tmp = np.array(np.squeeze(ncid.variables['__xarray_dataarray_variable__'][:]))
                    ncid.close()
                    data_mon = np.reshape(data_tmp, (int(data_tmp.shape[0]/24), 24, lat.shape[0], lat.shape[1]))                    
                    for th in range(len(flag_thresholds)):
                        wbgt[mm,:,th,:,:] = np.sum(data_mon-273.15 >= flag_thresholds[th], axis=0)
    
                # save the data
                np.savez(outfile,
                        lon = lon,
                        lat = lat,
                        flag_thresholds = flag_thresholds,
                        exceedances = wbgt[:,:,:,:,:])



# In[ ]:




