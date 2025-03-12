#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python


# # heatwave_an-maxlen.ipynb

# '''
#     File name: conditions_WBGT_monthmax.ipynb
#     Author: Andreas Prein
#     E-mail: prein@ucar.edu
#     Date created: 04.30.2024
#     Date last modified: 04.30.2024
# 
#     ############################################################## 
#     Needs data from:
#     
#     
#     Purpose:
# 
#     1) Read in WBGT data year by year
#     2) calculates annual heatwave frequency and maximum length and stores data for future processing
#     
# 
# '''

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
import random

import warnings
warnings.filterwarnings("ignore")


# In[3]:


def solar_time_difference_from_utc(longitude):
    from datetime import datetime, timedelta, timezone
    """
    Calculate the difference between local solar time and UTC based on the given longitude.

    Args:
    - longitude (float): The longitude of the location in degrees (East is positive, West is negative).

    Returns:
    - time_difference (timedelta): The difference from UTC as a timedelta object.
    """
    # Calculate the time difference due to longitude (in minutes)
    time_difference_minutes = longitude * 4  # 4 minutes per degree

    # Create a timedelta object for the time difference
    time_difference = timedelta(minutes=time_difference_minutes)
    
    return time_difference


# ### User imput section

# In[16]:


data_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData/GWBT/'
save_dir_base = '/glade/campaign/mmm/c3we/prein/Papers/2024/2024_WBGT-climate-change/heatwaves/'

simulation = "ctr" # ["ctr" "pgw"]

time_c404_ctr = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                              end=datetime.datetime(2021, 12, 31, 23), freq='h')
years_ctr = np.unique(time_c404_ctr.year)
conus404_pgw_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData_PGW/GWBT/'
time_c404_pgw = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                              end=datetime.datetime(2021, 12, 31, 23), freq='h')
years_pgw = np.unique(time_c404_pgw.year)


# In[5]:


temp_channge = np.arange(0,3.1,0.1)
delta_t = 0.25
minyears = 7


# ### Load global average LENS2 temperature for warming level selection

# In[6]:


data = pd.read_csv('/glade/u/home/prein/papers/2024/2024_WBGT_Climate-Change/data/NASA_global-av_t2m/graph.txt', delimiter=r"\s+", header=0, skiprows=[0,1,2,4])

StartDay = datetime.datetime(1850, 1, 15,1)
StopDay = datetime.datetime(2101, 1, 15,0)
Time=pd.date_range(StartDay, end=StopDay, freq='M')
Time_years=pd.date_range(StartDay, end=StopDay, freq='Y')
years = np.unique(Time.year)

reference_period = [1950, 1980]

lens2_t2m_dir = '/glade/campaign/mmm/c3we/mingge/DOD/LENS2/'
ncid=Dataset(lens2_t2m_dir + 'TREFMNAV_monthly_globalAvg_185001-197912.nc', mode='r')
TREFMNAV_hist = np.array(np.squeeze(ncid.variables['TREFMNAV'][:]))
ncid.close()
ncid=Dataset(lens2_t2m_dir + 'TREFMNAV_monthly_globalAvg_198001-201412.nc', mode='r')
TREFMNAV_cur = np.array(np.squeeze(ncid.variables['TREFMNAV'][:]))
ncid.close()
ncid=Dataset(lens2_t2m_dir + 'TREFMNAV_monthly_globalAvg_201501-210012.nc', mode='r')
TREFMNAV_fut = np.array(np.squeeze(ncid.variables['TREFMNAV'][:]))
ncid.close()

ncid=Dataset(lens2_t2m_dir + 'TREFMXAV_monthly_globalAvg_185001-197912.nc', mode='r')
TREFMXAV_hist = np.array(np.squeeze(ncid.variables['TREFMXAV'][:]))
ncid.close()
ncid=Dataset(lens2_t2m_dir + 'TREFMXAV_monthly_globalAvg_198001-201412.nc', mode='r')
TREFMXAV_cur = np.array(np.squeeze(ncid.variables['TREFMXAV'][:]))
ncid.close()
ncid=Dataset(lens2_t2m_dir + 'TREFMXAV_monthly_globalAvg_201501-210012.nc', mode='r')
TREFMXAV_fut = np.array(np.squeeze(ncid.variables['TREFMXAV'][:]))
ncid.close()

TREFMNAV = np.append(TREFMNAV_hist, TREFMNAV_cur, axis=1)
TREFMNAV = np.append(TREFMNAV, TREFMNAV_fut, axis=1)
TREFMXAV = np.append(TREFMXAV_hist, TREFMXAV_cur, axis=1)
TREFMXAV = np.append(TREFMXAV, TREFMXAV_fut, axis=1)
lens2_t2m = (TREFMNAV + TREFMXAV) / 2.

lens2_t2m_y = np.mean(np.reshape(lens2_t2m, (lens2_t2m.shape[0], int(lens2_t2m.shape[1]/12), 12)), axis=2)

ref_t2m = np.mean(lens2_t2m_y[:,(years >= reference_period[0]) & (years <= reference_period[1])])
time_to_t2m = np.mean(lens2_t2m_y, axis=0) - ref_t2m

dc_hist = np.arange(0,1.5,0.1)
dc_pgw = np.arange(0,3.5,0.1)
dc = 0.25 # deg. C

ref_period = [1950,1979]

era5_ref = np.mean(data['No_Smoothing'][(data['Year'] >= ref_period[0]) & (data['Year'] <= ref_period[1])])
era5_warming = np.squeeze(np.array([data['No_Smoothing'][data['Year'] == yy] - era5_ref for yy in range(1980,2022,1)]))

lens2_ref = np.mean(time_to_t2m[(years >= ref_period[0]) & (years <= ref_period[1])])
lens2_warming = np.array([time_to_t2m[years == yy][0] - lens2_ref for yy in range(2022,2022+len(np.unique(time_c404_pgw.year)),1)])

pgw_warming = np.zeros((len(years_ctr))); pgw_warming[:] = np.nan
for yy in range(len(years_ctr)):
    lens_hist = (years <= years_ctr[yy] + 5) & (years >= years_ctr[yy] - 5)
    lens_fut = (years <= years_ctr[yy] + 40 + 5) & (years >= years_ctr[yy] + 40 - 5)
    pgw_warming[yy] = data['No_Smoothing'][years_ctr[yy] == data['Year']] + np.mean(time_to_t2m[lens_fut]) - np.mean(time_to_t2m[lens_hist])


# In[7]:


warming_tar = 0.25
targ_ref = years_ctr[(era5_warming >= warming_tar - dc) & (era5_warming <= warming_tar + dc)]

warming_tar = 1
targ_1C = years_ctr[(era5_warming >= warming_tar - dc) & (era5_warming <= warming_tar + dc)]

warming_tar = 2
targ_2C = years_pgw[(pgw_warming[:len(years_pgw)] >= warming_tar - dc) & (pgw_warming[:len(years_pgw)] <= warming_tar + dc)]


# # Read CONUS404 CTR OR PGW WBGT DATA

# In[8]:


# read CONUS404 coordinates
ncid=Dataset('/glade/campaign/ncar/USGS_Water/CONUS404/wrfconstants_d01_1979-10-01_00:00:00.nc4', mode='r')
lon_conus = np.array(np.squeeze(ncid.variables['XLONG'][:]))
lat_conus = np.array(np.squeeze(ncid.variables['XLAT'][:]))
ncid.close()


# In[9]:


from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)
import cartopy.feature as cf
import cartopy.crs as ccrs
GEO_EM_D1 = '/glade/u/home/prein/projects/2020/2020_CONUS404/programs/plots/Domain/geo_em.d01.nc'
ncfile = Dataset(GEO_EM_D1)
HGT_M = getvar(ncfile, "HGT_M")
LU = getvar(ncfile, "LU_INDEX")
cart_proj = get_cartopy(HGT_M)
ncid.close()



# ### Process heat extreme data for cities

# In[39]:


# ================================================
# Heatwave length and frequency
def heatwave_stat(data):

    label_im, nb_labels = ndimage.label(data-273.15 >= 29)
    hw_feq = np.sum(np.unique(label_im, return_counts=True)[1][1:] >= 3) # !!! events longer or equal 3 days
    objects = scipy.ndimage.find_objects(label_im, axis=0)
    max_len = np.max([objects[ii][0].stop - objects[ii][0].start for ii in range(len(objects))])
    av_freq = hw_feq
    
    return max_len, av_freq


# In[58]:


from scipy.ndimage import label

lat_size = lon_conus.shape[0]
lon_size = lon_conus.shape[1]

if simulation == "ctr":
    time = time_c404_ctr
    data_dir_act = data_dir
else:
    time = time_c404_pgw
    data_dir_act = conus404_pgw_dir

    
years = np.unique(time.year)
for yy in tqdm(range(len(years))):
    save_file = save_dir_base + "/" + simulation + "_" + str(years[yy]) + "_heatwave-stats_v2.nc"
    if os.path.exists(save_file) == False:
        print("work on "+save_file)
        time_year = time[time.year == years[yy]]
        time_year_day = time_year[::24]
        data_year = np.zeros((int(len(time_year)/24), lon_conus.shape[0], lon_conus.shape[1])); data_year[:] = np.nan
        # # Load data
        for mm in tqdm(range(12)):
            data_ctr = xr.open_dataset(data_dir_act+"/GWBT_"+str(years[yy])+str(int(mm+1)).zfill(2)+"_CONUS404.nc")
            if simulation == "ctr":
                wbgt_act = data_ctr.GWBT.values
            else:
                wbgt_act = data_ctr.__xarray_dataarray_variable__.values
            wbgt_act = np.max(np.reshape(wbgt_act, (int(wbgt_act.shape[0]/24),24, lon_conus.shape[0], lon_conus.shape[1])), axis=1)
            data_year[time_year_day.month == int(mm+1),:,:] = wbgt_act
    
        # Initialize output arrays
        frequency = np.zeros((lat_size, lon_size), dtype=int)
        longest_period = np.zeros((lat_size, lon_size), dtype=int)
    
        exceed_mask = data_year-273.15 >= 29
        # Loop over each grid cell (lat, lon)
        for lat in range(lat_size):
            for lon in range(lon_size):
                # Get the time series for this grid point
                time_series = exceed_mask[:, lat, lon]

                label_im, nb_labels = ndimage.label(time_series)
                frequency[lat, lon] = np.sum(np.unique(label_im, return_counts=True)[1][1:] >= 3) # !!! events longer or equal 3 days
                objects = scipy.ndimage.find_objects(label_im)
                try:
                    longest_period[lat, lon] = np.max([objects[ii][0].stop - objects[ii][0].start for ii in range(len(objects))])
                except:
                    continue
        
        ds = xr.Dataset(
                {
                    "frequency": (["lat", "lon"], frequency),
                    "longest_period": (["lat", "lon"], longest_period),
                },
                coords={
                    # "lat": np.linspace(25, 50, lat_size),     # 1D latitudes
                    # "lon": np.linspace(-130, -60, lon_size),  # 1D longitudes
                    "lat_2d": (["lat", "lon"], lat_conus),  # 2D latitude grid
                    "lon_2d": (["lat", "lon"], lon_conus),  # 2D longitude grid
                },
            )

        # Save to a NetCDF file
        ds.to_netcdf(save_file)





