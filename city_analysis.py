#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# # city_analysis.ipynb

# '''
#     File name: conditions_WBGT_monthmax.ipynb
#     Author: Andreas Prein
#     E-mail: prein@ucar.edu
#     Date created: 04.30.2024
#     Date last modified: 04.30.2024
# 
#     ############################################################## 
#     Needs data from:
#     papers/2024/2024_WBGT_Climate-Change/programs/process_conditions_WBGT_monthmax/process_conditions_WBGT_monthmax.ipynb
#     papers/2024/2024_WBGT_Climate-Change/programs/conditions_WBGT/conditions_WBGT_monthmax-warming-targets.py
#     
#     Purpose:
# 
#     1) Read in preprocessed data
#     2) Calculate various statistics for specific regions (cities)
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


# ### User imput section

# In[ ]:


data_dir = '/glade/campaign/mmm/c3we/prein/Papers/2024/2024_WBGT-climate-change/WBGT_monmax_variables/'
city_data_dir = '/glade/campaign/mmm/c3we/prein/Papers/2024/2024_WBGT-climate-change/city_changes/'
save_dir = '/glade/campaign/mmm/c3we/prein/Papers/2024/2024_WBGT-climate-change/city_analysis/'

time_c404_ctr = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                              end=datetime.datetime(2019, 12, 31, 23), freq='h')
years_ctr = np.unique(time_c404_ctr.year)
conus404_pgw_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData_PGW/'
time_c404_pgw = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                              end=datetime.datetime(2011, 12, 31, 23), freq='h')
years_pgw = np.unique(time_c404_pgw.year)

cy = int(sys.argv[1])

cities = ['Phoenix',
          'Miami',
          'New Orleans',
          'Houston',
          'Atlanta',
          'DC',
          # 'Yuma',
          'Sacramento',
          'Dalas',
          'Charlotte',
          'Chicago',
          'New York City',
          'Tampa',
            ]

save_dir = save_dir + cities[cy] + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
city_loc = [
            [[350,370],[370,390]],
            [[1120,1135],[170,225]],
            [[860,880],[255,265]],
            [[730,760],[240,270]],
            [[980,1000],[373,400]],
            [[1110,1135],[540,575]],
            # [[292,300],[365,370]],
            [[180,195],[550,570]],
            [[685,720],[325,350]],
            [[1055,1070],[430,450]],
            [[875,900],[580,610]],
            [[1160,1180],[600,635]],
            [[1055,1070],[230,240]],
           ]

temp_channge = np.arange(0,3.1,0.1)
delta_t = 0.25
minyears = 7


# ### Load global average LENS2 temperature for warming level selection

# In[5]:


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
era5_warming = np.squeeze(np.array([data['No_Smoothing'][data['Year'] == yy] - era5_ref for yy in range(1980,2020,1)]))

lens2_ref = np.mean(time_to_t2m[(years >= ref_period[0]) & (years <= ref_period[1])])
lens2_warming = np.array([time_to_t2m[years == yy][0] - lens2_ref for yy in range(2022,2022+len(np.unique(time_c404_pgw.year)),1)])

pgw_warming = np.zeros((len(years_ctr))); pgw_warming[:] = np.nan
for yy in range(len(years_ctr)):
    lens_hist = (years <= years_ctr[yy] + 5) & (years >= years_ctr[yy] - 5)
    lens_fut = (years <= years_ctr[yy] + 40 + 5) & (years >= years_ctr[yy] + 40 - 5)
    pgw_warming[yy] = data['No_Smoothing'][years_ctr[yy] == data['Year']] + np.mean(time_to_t2m[lens_fut]) - np.mean(time_to_t2m[lens_hist])


# In[6]:


warming_tar = 0.25
targ_years_ref = (era5_warming >= warming_tar - dc) & (era5_warming <= warming_tar + dc)

warming_tar = 2
targ_years = (pgw_warming[:len(years_pgw)] >= warming_tar - dc) & (pgw_warming[:len(years_pgw)] <= warming_tar + dc)


# # Read CONUS404 CTR OR PGW WBGT DATA

# In[7]:


# read CONUS404 coordinates
ncid=Dataset('/glade/campaign/ncar/USGS_Water/CONUS404/wrfconstants_d01_1979-10-01_00:00:00.nc4', mode='r')
lon_conus = np.array(np.squeeze(ncid.variables['XLONG'][:]))
lat_conus = np.array(np.squeeze(ncid.variables['XLAT'][:]))
ncid.close()


# In[8]:


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




# ### Load hourly data over city and derive statistics

# In[70]:


datasets = ['CONUS404_ctr', 'CONUS404_pgw']
flag_thresholds = [29, 31, 32]

lat_cy = lat_conus[city_loc[cy][1][0]:city_loc[cy][1][1],
                   city_loc[cy][0][0]:city_loc[cy][0][1]]
lon_cy = lon_conus[city_loc[cy][1][0]:city_loc[cy][1][1],
                   city_loc[cy][0][0]:city_loc[cy][0][1]]


# In[71]:


for da in range(len(datasets)):
    print('WORK ON '+datasets[da])
    outfile = save_dir + datasets[da] + '_'+cities[cy]+'_statistics.nc'
    if os.path.isfile(outfile) == False:
        if datasets[da] == 'CONUS404_ctr':
            data_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData/GWBT/'
            time = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                                  end=datetime.datetime(2021, 12, 31, 23), freq='h')
            data_cy = np.zeros((len(time), lat_cy.shape[0], lat_cy.shape[1]))
            years = np.unique(time.year)
            for yy in tqdm(range(len(years))):
                for mm in range(12):
                    time_act = (years[yy] == time.year) & (mm+1 == time.month)
                    file = data_dir + 'GWBT_'+str(years[yy])+str(mm+1).zfill(2)+'_CONUS404.nc'
                    ncid=Dataset(file, mode='r')
                    data_cy[time_act,:] = np.array(np.squeeze(ncid.variables['GWBT'][:,city_loc[cy][1][0]:city_loc[cy][1][1],
                                                                             city_loc[cy][0][0]:city_loc[cy][0][1]]))
                    ncid.close()
    
        elif datasets[da] == 'CONUS404_pgw':
            data_dir = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData_PGW/GWBT/'
            time = pd.date_range(datetime.datetime(1980, 1, 1, 0), 
                                  end=datetime.datetime(2021, 12, 31, 23), freq='h')
            data_cy = np.zeros((len(time), lat_cy.shape[0], lat_cy.shape[1]))
            years = np.unique(time.year)
            for yy in tqdm(range(len(years))):
                for mm in range(12):
                    time_act = (years[yy] == time.year) & (mm+1 == time.month)
                    file = data_dir + 'GWBT_'+str(years[yy])+str(mm+1).zfill(2)+'_CONUS404.nc'
                    ncid=Dataset(file, mode='r')
                    data_cy[time_act,:] = np.array(np.squeeze(ncid.variables['__xarray_dataarray_variable__'][:,city_loc[cy][1][0]:city_loc[cy][1][1],
                                                                             city_loc[cy][0][0]:city_loc[cy][0][1]]))
                    ncid.close()
    
                

    ds = xr.Dataset(
        {
            "WBGT": (["time", "y", "x"], data_cy)
        },
        coords={
            "time": time,
            "y": np.arange(lat_cy.shape[0]),
            "x": np.arange(lat_cy.shape[1]),
            "latitude": (["y", "x"], lat_cy),
            "longitude": (["y", "x"], lon_cy)
        }
    )
    
    # Set attributes (optional but recommended)
    ds.attrs['long_name'] = 'Wet bulb globe temperature dimensions time, latitude, longitude'
    ds['WBGT'].attrs['units'] = 'K'
    
    # Write to a NetCDF file
    ds.to_netcdf(outfile)
    
    print(f"Data successfully written to {outfile}")





