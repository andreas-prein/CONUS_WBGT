#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# # nighttime_minima.ipynb

# In[2]:


'''
    File name: nighttime_minima.ipynb
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 04.07.2024
    Date last modified: 04.07.2024

    ############################################################## 
    Purpose:

    Read in WBGT data from CONUS404 control and PGW that were processed in:
    projects/2020_CONUS404/programs/CONUS404_preprocessor/GWBT_code/PyWBGT-1.0.0/Jupyter_notebooks/Calculate_WBGT_with_C404_data.py

    1) Calculate yearly maxima nigthtime minima WBGTs
    2) Calculate exceedance frequencies of green, yellow, red, and black flag hour exceedances from 8 pm to 6 am

    Save the data to npz files for further processing.
    
'''


# In[3]:


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


# In[4]:


years = [int(sys.argv[1])]

datasets = ['ERA5', 'CONUS404_ctr', 'CONUS404_pgw']
# datasets = ['CONUS404_ctr', 'CONUS404_pgw']

flag_thresholds = [26, 29, 31, 32]

savedir = '/glade/campaign/mmm/c3we/prein/Papers/2024/2024_WBGT-climate-change/nightime_stats/'


# In[5]:


import numpy as np

def compute_max_nighttime_min_temp(temperature, longitude_grid, threshold=25):
    """
    Computes the maximum of the daily nighttime minimum temperatures and threshold exceedances.
    
    Parameters:
    - temperature (numpy array): 3D array of shape (time, lat, lon) containing temperature data (°C).
    - longitude_grid (numpy array): 2D array of shape (lat, lon) with longitude values (degrees).
    - threshold (float): Temperature threshold (°C) for exceedance count. Default is 25°C.
    
    Returns:
    - max_nighttime_min_temp (numpy array): 2D array (lat, lon) of maximum nighttime minimum temperatures.
    - exceedance_count (numpy array): 2D array (lat, lon) of the number of nighttime exceedances.
    """
    # Extract dimensions
    time_steps, lat_size, lon_size = temperature.shape

    # Ensure the dataset spans whole days (should be 31 days x 24 hours = 744)
    assert time_steps % 24 == 0, "Time dimension should be a multiple of 24 (full days)."

    # Compute UTC hours
    utc_hours = np.arange(time_steps) % 24  # Extract UTC hour

    # Reshape longitude grid for broadcasting (lat, lon -> 1, lat, lon)
    longitude_grid = longitude_grid[None, :, :]  # Shape (1, lat, lon)

    # Compute Local Solar Time (LST)
    lst_hours = (utc_hours[:, None, None] + longitude_grid / 15) % 24  # Shape (time, lat, lon)

    # Define nighttime hours (8 PM - 6 AM LST)
    nighttime_mask = (lst_hours >= 20) | (lst_hours < 6)

    # Apply nighttime mask: keep only nighttime temperatures, set others to NaN
    temperature_nighttime = np.where(nighttime_mask, temperature, np.nan)

    # Reshape into (days, 24, lat, lon) format
    daily_temps = temperature_nighttime.reshape(-1, 24, lat_size, lon_size)

    # Compute minimum nighttime temperature for each day
    daily_nighttime_min = np.nanmin(daily_temps, axis=1)  # Min over 24 hours

    # Compute the maximum of these daily nighttime minima
    max_nighttime_min_temp = np.nanmax(daily_nighttime_min, axis=0)

    # Compute exceedance count: count nighttime hours exceeding the threshold
    exceedance_count = np.nansum(temperature_nighttime > threshold, axis=0)

    return max_nighttime_min_temp, exceedance_count


# ### loop over datasets and do the processing

# In[ ]:


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
        lon2D,lat2D = np.meshgrid(lon,lat)

        # years = np.unique(time.year)
        for yy in tqdm(range(len(years))):
            outfile = savedir + str(years[yy]) + '_' + datasets[da] + '_nightime_stats.npz'
            if os.path.isfile(outfile) == False:
                wbgt_night_max = np.zeros((12, len(flag_thresholds), len(lat), len(lon))); wbgt_night_max[:] = np.nan
                night_exceedance = np.zeros((12, len(flag_thresholds), len(lat), len(lon))); night_exceedance[:] = np.nan
                
                file = data_dir + 'WBGT_ERA5_'+str(years[yy])+'.nc'
                ncid=Dataset(file, mode='r')
                data_tmp = np.array(np.squeeze(ncid.variables['WBGT'][:,conus_era5[0]:conus_era5[2],conus_era5[3]:conus_era5[1]]))
                ncid.close()
                
                for mm in range(12):
                    data_mon = data_tmp[time[time.year == years[yy]].month == (mm+1)]
                    for th in range(len(flag_thresholds)):
                        wbgt_night_max[mm,th,:,:], night_exceedance[mm,th,:,:] = compute_max_nighttime_min_temp(data_mon, lon2D, threshold= flag_thresholds[th]+273.15)
    
                # save the data
                np.savez(outfile,
                        lon = lon,
                        lat = lat,
                        flag_thresholds = flag_thresholds,
                        wbgt_night_max = wbgt_night_max,
                        night_exceedance = night_exceedance
                        )
            
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
            wbgt_night_max = np.zeros((12, len(flag_thresholds), lat.shape[0], lat.shape[1])); wbgt_night_max[:] = np.nan
            night_exceedance = np.zeros((12, len(flag_thresholds), lat.shape[0], lat.shape[1])); night_exceedance[:] = np.nan
            outfile = savedir + str(years[yy]) + '_' + datasets[da] + '_nightime_stats.npz'
            if os.path.isfile(outfile) == False:
                
                for mm in tqdm(range(12)):
                    file = data_dir + 'GWBT_'+str(years[yy])+str(mm+1).zfill(2)+'_CONUS404.nc'
                    ncid=Dataset(file, mode='r')
                    data_mon = np.array(np.squeeze(ncid.variables['GWBT'][:]))
                    ncid.close()
                    for th in range(len(flag_thresholds)):
                        wbgt_night_max[mm,th,:,:], night_exceedance[mm,th,:,:] = compute_max_nighttime_min_temp(data_mon, lon, threshold= flag_thresholds[th]+273.15)
    
                # save the data
                np.savez(outfile,
                        lon = lon,
                        lat = lat,
                        flag_thresholds = flag_thresholds,
                        wbgt_night_max = wbgt_night_max,
                        night_exceedance = night_exceedance
                        )
        
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
            wbgt_night_max = np.zeros((12, len(flag_thresholds), lat.shape[0], lat.shape[1])); wbgt_night_max[:] = np.nan
            night_exceedance = np.zeros((12, len(flag_thresholds), lat.shape[0], lat.shape[1])); night_exceedance[:] = np.nan
            outfile = savedir + str(years[yy]) + '_' + datasets[da] + '_nightime_stats.npz'
            if os.path.isfile(outfile) == False:
                for mm in tqdm(range(12)):
                    file = data_dir + 'GWBT_'+str(years[yy])+str(mm+1).zfill(2)+'_CONUS404.nc'
                    ncid=Dataset(file, mode='r')
                    data_mon = np.array(np.squeeze(ncid.variables['__xarray_dataarray_variable__'][:]))
                    ncid.close()
                    for th in range(len(flag_thresholds)):
                        wbgt_night_max[mm,th,:,:], night_exceedance[mm,th,:,:] = compute_max_nighttime_min_temp(data_mon, lon, threshold= flag_thresholds[th]+273.15)
    
                # save the data
                np.savez(outfile,
                        lon = lon,
                        lat = lat,
                        flag_thresholds = flag_thresholds,
                        wbgt_night_max = wbgt_night_max,
                        night_exceedance = night_exceedance
                        )



# In[ ]:


th=0
wbgt_night_max[mm,th,:,:], wbgt_night_max[mm,th,:,:] = compute_max_nighttime_min_temp(data_mon, lon2D, threshold= flag_thresholds[th])


# In[ ]:


plt.pcolormesh(exceed)


# In[ ]:


lon2D,lat2D = np.meshgrid(lon,lat)


# In[ ]:




