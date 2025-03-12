#!/usr/bin/env python
'''
    File name: date_range
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.01.2015
    Date last modified: 05.01.2015

    ##############################################################
    Purpos:
    This function creates a time vector between two given dates

    inputs:
    start_date
    end_date
    increment --> how many time steps do we have per day
    period       --> 'days','hours',...

    returns:
    rgrtime --> time vector
'''


from datetime import datetime
from dateutil.relativedelta import relativedelta
from pdb import set_trace as stop

def date_range(start_date, end_date, increment):
    result = []
    nxt = start_date
    delta = relativedelta(**{'days':increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result





'''
    File name: TemporalFrequency
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 09.01.2015
    Date last modified: 17.07.2015

    ##############################################################
    Purpos:

    gets the daily frequency of a data set (e.g., 24 for hourly, 1 for
    daily, ~0.03 for monthly) and returns a string according to the
    frequency (e.g., 'H' for hourly)

    inputs:
    rFrequency

    returns:
    sTempFreq

    history:

    2015 07 17    round frequency to 2 coma digits first
'''

def TemporalFrequency(rFrequency):

    import numpy as np

    if round(rFrequency) == 24:
        sTempFreq='H'
    elif round(rFrequency) == 8:
        sTempFreq='3H'
    elif round(rFrequency) == 3:
        sTempFreq='8H'
    elif round(rFrequency,1) == 2:
        sTempFreq='12H'
    elif round(rFrequency,1) == 1:
        sTempFreq='D'
    elif round(rFrequency, 2) == 1/30.:
        sTempFreq='M'
    elif round(rFrequency, 4) == 1/365.:
        sTempFreq='Y'
    else:
        print( '    Temporal frequency not defined jet!')
        stop()


    return sTempFreq







'''
    File name: fnCDOregrid
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 11.01.2015
    Date last modified: 11.01.2015

    ##############################################################
    Purpos:

    This function remapps data from a source grid to a target grid using
    the CDO remapping routines. Therefore, the source grid data and the
    source grid is saved as netcdf. Afterwards the CDO is applied to the
    netcdfs and the remapped data is read back into python.

    returns:
    dfDataTarget --> dataframe with data on target grid
'''

def fnCDOregrid(dfDataSource,
                rgrLonSource,
                rgrLatSource,
                dfDataDest,
                rgrLonTarget,
                rgrLatTarget,
                sRemapMeth='remapcon'):

    #from cdo import *   # python version
    #cdo = Cdo()
    from netCDF4 import Dataset
    from pdb import set_trace as stop
    import numpy as np
    import pandas as pd
    import subprocess
    import os

    iIterationLimit=200000000000. # 20000000000.

    # check if the data is too big and if we have to iterate the remapping
    import operator
    import functools
    iElements=functools.reduce(operator.mul, dfDataSource.shape, 1)
    import math
    iIterations=math.ceil(iElements/iIterationLimit)
    # Split the data if nescesarry
    rgiIterate=[0]*int(iIterations+1)
    rgiIterate[-1]=dfDataSource.shape[0]
    for it in range(int(iIterations)):
        if it < iIterations:
            rgiIterate[it]=round(it*(dfDataSource.shape[0]/iIterations))


    rgrDataDest=np.empty((dfDataSource.shape[0],rgrLonTarget.shape[0],rgrLonTarget.shape[1]))
    rgrDataDest[:] = np.NAN

    # generate a unique directory for the remapping
    iFile=1
    sTempFolder='/glade/derecho/scratch/prein/tmp'
    while (os.path.isdir(sTempFolder+str(iFile)) == 1):
        iFile=iFile+1
    else:
        sTempFolder=sTempFolder+str(iFile)
        subprocess.call(["mkdir","-p",sTempFolder])
    # loop over the desired iterations
    for it in range(int(iIterations)):
        rgrData=dfDataSource[rgiIterate[it]:rgiIterate[it+1]]

        # if we want to remap conservatively we have to generate the grid cell
        # coners firs
        if sRemapMeth == 'remapcon' or sRemapMeth == 'remapcon2' or \
                sRemapMeth == 'remaplaf' or sRemapMeth == 'remapbic':
            from HelperFunctions import fnCellBoundaries
            rgrLonBNDSource,rgrLatBNDSource=fnCellBoundaries(rgrLonSource,
                                                             rgrLatSource)
            rgrLonBNDTarget,rgrLatBNDTarget=fnCellBoundaries(rgrLonTarget,
                                                             rgrLatTarget)

        # proceed with page 10f in https://www.rsmas.miami.edu/users/rajib/cdo.pdf
        # ___________________________________________________________________
        # Start wit simple remapping routines
        if sRemapMeth == 'remapbil' or sRemapMeth == 'remapnn' or \
                sRemapMeth == 'remapdis':
            for mo in range(0, 2):
                if mo == 0:
                    # source grid
                    rgrData=dfDataSource[rgiIterate[it]:rgiIterate[it+1],:,:]
                    rgiSize=rgrData.shape
                    rgrLat=rgrLatSource
                    rgrLon=rgrLonSource
                    sFile=sTempFolder+'/SourceData.nc'
                if mo == 1:
                    # target grid
                    rgrData=dfDataDest[rgiIterate[it]:rgiIterate[it+1],:,:]
                    rgiSize=rgrData.shape
                    rgrLat=rgrLatTarget
                    rgrLon=rgrLonTarget
                    sFile=sTempFolder+'/TargetGrid.nc'
                root_grp = Dataset(sFile, 'w', format='NETCDF4')
                # dimensions
                root_grp.createDimension('Time', None)
                root_grp.createDimension('X', rgiSize[1])
                root_grp.createDimension('Y', rgiSize[2])
                # variables
                temp = root_grp.createVariable('Data', 'f4', ('Time', 'X','Y',))
                lat = root_grp.createVariable('XLAT', 'f4', ('X','Y',))
                lon = root_grp.createVariable('XLON', 'f4', ('X','Y',))
                # Variable Attributes
                temp.coordinates = "XLON XLAT"
                lat.units = 'degree_north'
                lon.units = 'degree_east'
                # write data to netcdf
                lat[:]=rgrLat
                lon[:]=rgrLon
                temp[:]=rgrData
                root_grp.close()
                subprocess.call(["cdo", "-L", sRemapMeth+","+sTempFolder+"/TargetGrid.nc", sTempFolder+"/SourceData.nc", sTempFolder+"/TargetData.nc"])
        else:
            # For these remapping methods we need the grid cell boundaries!
            # Writing source and target grid as well as source data netcdf file
            for mo in range(0, 3):
                if mo == 0:
                    # source grid
                    rgiSize=dfDataSource.shape
                    rgrLatBND=rgrLatBNDSource
                    rgrLonBND=rgrLonBNDSource
                    rgrLat=rgrLatSource
                    rgrLon=rgrLonSource
                    sFile=sTempFolder+'/src_grid.nc'
                if mo == 1:
                    # target grid
                    rgiSize=dfDataDest.shape #(1,rgrLonTarget.shape[0],rgrLonTarget.shape[1])
                    rgrLatBND=rgrLatBNDTarget
                    rgrLonBND=rgrLonBNDTarget
                    rgrLat=rgrLatTarget
                    rgrLon=rgrLonTarget
                    sFile=sTempFolder+'/tar_grid.nc'
                if mo == 2:
                    # source data
                    rgrData=dfDataSource[rgiIterate[it]:rgiIterate[it+1],:,:]
                    rgiSize=rgrData.shape
                    rgrLatBND=rgrLatBNDSource
                    rgrLonBND=rgrLonBNDSource
                    rgrLat=rgrLatSource
                    rgrLon=rgrLonSource
                    sFile=sTempFolder+'/SourceData.nc'

                root_grp = Dataset(sFile, 'w', format='NETCDF4')
                # dimensions
                if mo != 2: root_grp.createDimension('grid_size',rgiSize[2]*rgiSize[1])
                if mo == 2: root_grp.createDimension('Time', None)
                root_grp.createDimension('grid_ysize', rgiSize[1])
                root_grp.createDimension('grid_xsize', rgiSize[2])
                if mo != 2: root_grp.createDimension('grid_corners',4)
                if mo != 2: root_grp.createDimension('grid_rank',2)
                # variables
                if mo != 2: grid_dim =  root_grp.createVariable('grid_dims','i',('grid_rank',))
                if mo == 2: temp = root_grp.createVariable('Data', 'f8', ('Time', 'grid_ysize','grid_xsize',))
                lat = root_grp.createVariable('grid_center_lat', 'f8', ('grid_ysize','grid_xsize',))
                lon = root_grp.createVariable('grid_center_lon', 'f8', ('grid_ysize','grid_xsize',))
                if mo != 2: grid_imask = root_grp.createVariable('grid_imask','i',('grid_ysize','grid_xsize',))
                if mo != 2: grid_corner_lat = root_grp.createVariable('grid_corner_lat','f8',('grid_ysize','grid_xsize','grid_corners',))
                if mo != 2: grid_corner_lon = root_grp.createVariable('grid_corner_lon','f8',('grid_ysize','grid_xsize','grid_corners',))
                if mo == 2: time = root_grp.createVariable('time','f8',('Time'))
                # Variable Attributes
                lat.units = 'degrees'
                lon.units = 'degrees'
                if mo == 2:
                    temp.coordinates = "grid_center_lon grid_center_lat"
                    time.long_name = "time"
                    time.units = "day since 1978-01-01 12:00:00"
                    time.calendar = "gregorian"
                # temp.bounds = "grid_corner_lon grid_corner_lat"
                if mo != 2:
                    grid_corner_lat.units="degrees"
                    grid_corner_lon.units="degrees"
                    lon.bounds= "grid_corner_lon"
                    lat.bounds= "grid_corner_lat"
                    grid_imask.units = "unitless"
                    grid_imask.coordinates = "grid_center_lon grid_center_lat"
                # write data to netcdf
                lat[:]=rgrLat
                lon[:]=rgrLon
                if mo == 2:
                    temp[:]=rgrData
                    time[:]=range(rgiSize[0])
                if mo != 2:
                    grid_corner_lat[:]=rgrLatBND
                    grid_corner_lon[:]=rgrLonBND
                    grid_dim[:]=[rgiSize[2],rgiSize[1]]
                    grid_imask[:]=np.zeros((rgiSize[1],rgiSize[2]))[:]=1
                root_grp.close()
            iTest=-1
            for tt in range(0,100):
                # this loop is unfortunably nescessary because these remapping
                # methods do sometimes not work on yellowstone
                # subprocess.call(["bsub", "-q", "geyser", "-W", "1:30", "-P", "P66770001", "-n", "1", "cdo", "-L", sRemapMeth+",/glade/scratch/prein/tmp/tar_grid.nc", "-selvar,Data", "-setgrid,/glade/scratch/prein/tmp/src_grid.nc", sTempFolder+"/SourceData.nc", sTempFolder+"/TargetData.nc"])
                # iTest=subprocess.call(["bsub", "<", "cdo", "-L", sRemapMeth+",/glade/scratch/prein/tmp/tar_grid.nc", "-selvar,Data", "-setgrid,/glade/scratch/prein/tmp/src_grid.nc", sTempFolder+"/SourceData.nc", sTempFolder+"/TargetData.nc"])
#                 stop()
#                 pp = subprocess.Popen("cdo -L "+sRemapMeth+","+sTempFolder+"/tar_grid.nc -selvar,Data -setgrid,"+sTempFolder+"/src_grid.nc "+ sTempFolder+"/SourceData.nc "+ sTempFolder+"/TargetData.nc", shell=True)
#                 pp.wait()
                iTest=subprocess.call(["cdo", "-L", sRemapMeth+","+sTempFolder+"/tar_grid.nc", "-selvar,Data", "-setgrid,"+sTempFolder+"/src_grid.nc", sTempFolder+"/SourceData.nc", sTempFolder+"/TargetData.nc"])
                print( '         Interating Remapping: '+str(tt))
                if iTest == 0:
                    break
                else:
                    subprocess.call(['rm',sTempFolder+"/TargetData.nc"])

        # now we can read in the remapped data from TargetData.nc
        ncid=Dataset(sTempFolder+"/TargetData.nc", mode='r') # open the netcdf
        rgrDataTarget=np.squeeze(ncid.variables["Data"][:])

        # set missing values to np.nan
        vars=ncid.variables
        if '_MissingValue' in (vars["Data"]).ncattrs():
            rNaNval=ncid.variables["Data"]._MissingValue
            rgrDataTarget[np.array(rgrDataTarget)==np.array(rNaNval)] = np.nan
        elif '_FillValue' in (vars["Data"]).ncattrs():
            rNaNval=ncid.variables["Data"]._FillValue
            rgrDataTarget[np.array(rgrDataTarget)==np.array(rNaNval)] = np.nan
        ncid.close()

        # claen up now
        subprocess.call(["rm","-rf",sTempFolder])

        # Concanate the remapped time slizes
        rgrDataDest[rgiIterate[it]:rgiIterate[it+1],:,:]=rgrDataTarget

    # generate pannel
    dfData = rgrDataDest

    return dfData



'''
    File name: fnCellBoundaries
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 11.01.2015
    Date last modified: 11.01.2015

    ##############################################################
    Purpos:

    This function estimates the grid cell boundaries from a matrix of
    center points. It is assuming that the grid cell boundaries are
    straight lines and that there are only four corners!

    returns:
    rgrLonBNDSource
    rgrLatBNDSource --> grid cell boundaries in the format ( grid ysize , grid xsize , grid corners )
'''

def fnCellBoundaries(rgrLon,rgrLat):
    import numpy as np
    from pdb import set_trace as stop

    rgiSize=rgrLon.shape
    rgrLonBND=np.zeros((rgiSize[0]+1,rgiSize[1]+1))
    rgrLatBND=np.zeros((rgiSize[0]+1,rgiSize[1]+1))
    for lo in range(0, rgiSize[0]+1):
        for la in range(0, rgiSize[1]+1):
            if lo < rgiSize[0]-1 and la < rgiSize[1]-1:
                #All points except at the boundaries
                rgrLonBND[lo, la] = rgrLon[lo, la]-(rgrLon[lo+1, la+1]-rgrLon[lo, la])/2
                rgrLatBND[lo, la] = rgrLat[lo, la]-(rgrLat[lo+1, la+1]-rgrLat[lo, la])/2
            if lo >= rgiSize[0]-1 and la < rgiSize[1]-1:
                #reight boundary second last row
                rgrLonBND[lo, la] = rgrLon[lo-1, la]+(rgrLon[lo-1, la]-rgrLon[lo-2, la+1])/2
                rgrLatBND[lo, la] = rgrLat[lo-1, la]-(rgrLat[lo-2, la+1]-rgrLat[lo-1, la])/2
            if lo < rgiSize[0]-1 and la >= rgiSize[1]-1:
                #upper boundary second last row
                rgrLonBND[lo, la] = rgrLon[lo, la-1]-(rgrLon[lo+1, la-2]-rgrLon[lo, la-1])/2
                rgrLatBND[lo, la] = rgrLat[lo, la-1]-(rgrLat[lo+1, la-2]-rgrLat[lo, la-1])/2
            if lo >= rgiSize[0]-1 and la >= rgiSize[1]-1:
                #upper right grid cells
                rgrLonBND[lo, la] = rgrLon[lo-1, la-1]-(rgrLon[lo-2, la-2]-rgrLon[lo-1, la-1])/2
                rgrLatBND[lo, la] = rgrLat[lo-1, la-1]-(rgrLat[lo-2, la-2]-rgrLat[lo-1, la-1])/2

    #we have to bring the result in a different format
    rgrLonBND2=np.zeros((rgiSize[0],rgiSize[1],4))
    rgrLatBND2=np.zeros((rgiSize[0],rgiSize[1],4))

    rgrLonBND2[:,:,0]=rgrLonBND[:-1,:-1]
    rgrLonBND2[:,:,1]=rgrLonBND[:-1,1:]
    rgrLonBND2[:,:,2]=rgrLonBND[1:,1:]
    rgrLonBND2[:,:,3]=rgrLonBND[1:,:-1]

    rgrLatBND2[:,:,0]=rgrLatBND[:-1,:-1]
    rgrLatBND2[:,:,1]=rgrLatBND[:-1,1:]
    rgrLatBND2[:,:,2]=rgrLatBND[1:,1:]
    rgrLatBND2[:,:,3]=rgrLatBND[1:,:-1]

    return rgrLonBND2,rgrLatBND2




'''
    File name: fnMapSideRatio
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 03.03.2015
    Date last modified: 03.03.2015

    ##############################################################
    Purpos:

    This function calculates the ratio of the X verses the Y side lenght of
    a map according to the lenght between the largest north-south and
    east-west distance.

    returns:
    rgrXYratio
'''

def fnMapSideRatio(rgrlon,  #lon coordinates 2D
                   rgrlat):              #lat coordinates 2D

    import numpy as np
    from pdb import set_trace as stop
    from math import sin, cos, sqrt, atan2, radians

    from HelperFunctions import fnCoordDist

    rgiSize=rgrlon.shape
    rgrDistNS1=fnCoordDist((rgrlon[0,0],rgrlat[0,0]),
                           (rgrlon[rgiSize[0]-1,0],rgrlat[rgiSize[0]-1,0]))
    rgrDistNS2=fnCoordDist((rgrlon[0,rgiSize[1]-1],rgrlat[0,rgiSize[1]-1]),
                           (rgrlon[rgiSize[0]-1,rgiSize[1]-1],rgrlat[rgiSize[0]-1,rgiSize[1]-1]))
    rgrDistEW1=fnCoordDist((rgrlon[0,0],rgrlat[0,0]),
                           (rgrlon[0,rgiSize[1]-1],rgrlat[0,rgiSize[1]-1]))
    rgrDistEW2=fnCoordDist((rgrlon[rgiSize[0]-1,0],rgrlat[rgiSize[0]-1,0]),
                           (rgrlon[rgiSize[0]-1,rgiSize[1]-1],rgrlat[rgiSize[0]-1,rgiSize[1]-1]))

    rgrXYratio=np.max([rgrDistEW1,rgrDistEW2])/np.max([rgrDistNS1,rgrDistNS2])

    return rgrXYratio





'''
    File name: fnCoordDist
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 03.03.2015
    Date last modified: 03.03.2015

    ##############################################################
    Purpos:

    Calculates the distance (in km) between two coordinates [lat,lon]
    in degrees.

    returns:
    rDistance
'''

def fnCoordDist(rgrLonLat1,
                rgrLonLat2):
    from math import sin, cos, sqrt, atan2, radians

    R = 6373.0 #Earth radius in km
    lat1 = radians(rgrLonLat1[1])
    lon1 = radians(rgrLonLat1[0])
    lat2 = radians(rgrLonLat2[1])
    lon2 = radians(rgrLonLat2[0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    rDistance = R * c

    return rDistance






'''
    File name: fnSubregion
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.03.2015
    Date last modified: 05.03.2015

    ##############################################################
    Purpos:

    This function gives back the indices of a subregion corresponding to a
    two dimensional lat lon coordinate system based on a region defined as [N,E,S,W]

    returns:
    rgrDomainOffsets --> array with the offsets for the subregion
'''


def fnSubregion(rgrLon,
                rgrLat,
                rgrSubRegion=None,     #array which defines the subregion as [N,E,S,W] (only use if rgiFinite is not defined!)
                rgiFinite=None):            # matrix which has 1/TRUE values in the region which should be cut out (only use if rgrSubRegion is not defined!)

    import numpy as np

    if rgrSubRegion != None:
        #generate a matrix with ones in the subregion and zeros elsewehere
        rgiSubRegion=np.array((rgrLon >= rgrSubRegion[3]) & (rgrLon <= rgrSubRegion[1]) & (rgrLat <= rgrSubRegion[0]) & (rgrLat >= rgrSubRegion[2]))


    if sum(sum(rgiSubRegion)) == 0:
        print( '        The suggested subregion could not be cut out! Stop the program now.')
        stop()

    rgiSize=rgrLon.shape
    iEastBound=rgiSize[0]
    iWestBound=0
    iNorthBound=0
    iSouthBound=rgiSize[1]
    for lon in range(0, rgiSize[1]):
        if sum(sum([rgiSubRegion[:,lon]])) > 0:
            iWestBound=lon
            break
    for lo in range(rgiSize[1]-1, 0, -1):
        if sum(sum([rgiSubRegion[:,lo]])) > 0:
            iEastBound=lo
            break
    for la in range(0, rgiSize[0]):
        if sum(sum([rgiSubRegion[la,:]])) > 0:
            iSouthBound=la
            break
    for lat in range(rgiSize[0]-1, 0, -1):
        if sum(sum([rgiSubRegion[lat,:]])) > 0:
            iNorthBound=lat
            break
    rgrDomainOffsets=[iNorthBound,iEastBound,iSouthBound,iWestBound]


    return rgrDomainOffsets





#!/usr/bin/env python
'''
    File name: fnSetComNaN
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 13.03.2015
    Date last modified: 13.03.2015

    ##############################################################
    Purpos:

    This function sets the same elements (ReplVal; Def=np.nan) to the
    provided nanValSet value (default=np.nan) in all datasets in an
    dictionary. The datasets have to have the same dimensions.


    returns:
    dcData --> dictionary with all datasets
'''
import numpy as np
from pdb import set_trace as stop
import pandas as pd

def fnSetComNaN(
    dcData,                 # dictionarry with data sets
    ):

    rgsDatasets=dcData.keys()
    # loop over data sets and generate a common NaN matrix
    ii=0
    rgrNaNAct=[]
    for da in rgsDatasets:
        if ii == 0:
            rgrNaN=np.isnan(np.asarray(dcData[da]))
        else:
            rgrNaN=rgrNaN+np.isnan(np.asarray(dcData[da]))
        ii=ii+1

    # loop over the data sets and set common NaN values
    for da in rgsDatasets:
        rgrDatAct=np.array(dcData[da])
        rgrDatAct[np.array(rgrNaN)]=np.nan
        if str(type(dcData[da])) == "<class 'pandas.core.panel.Panel'>":
            dcData[da]=pd.Panel(rgrDatAct,items=dcData[da].items,  major_axis=dcData[da].major_axis, minor_axis=dcData[da].minor_axis)
        else:
            dcData[da]=rgrDatAct

    return dcData




#!/usr/bin/env python
'''
    File name: fnShapeFile
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 18.03.2015
    Date last modified: 18.03.2015

    ##############################################################
    Purpos:

    This function reads in ESRI conform shape files and stores the ons
    provided by dcSRs in a dictionary.


    returns:
    dcShapeFiles --> dictionary with shapefiles
'''

def fnShapeFile(
    dcSRs):                  # dictionarry with an dictionary entry
                             # SubRegDef which keys contain the names of
                             # the shapefiles stored in an entry SubRegFile

    import sys, traceback
    import dateutil.parser as dparser
    import string
    from pdb import set_trace as stop
    import numpy as np
    import shapefile

    for sr in (dcSRs['SubRegDef']).keys():
        sShapeFiles=dcSRs['SubRegFile']
        sf = shapefile.Reader(sShapeFiles+sr)
        rgrShapes = sf.shapes()[0].points[:]
        dcSRs['SubRegDef'][sr]=[dcSRs['SubRegDef'][sr][:],rgrShapes]

    return dcSRs




#!/usr/bin/env python
'''
    File name: rgrSRmask
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 19.03.2015
    Date last modified: 19.03.2015

    ##############################################################
    Purpos:

    This function reads shape file polygons derifed by a prviouse call of
    fnShapeFile and maskes a provided lat lon grid with them. The proceture
    only checks if the center of a grid cell is in the polygon not if more
    than the half of the grid cell is covered.


    returns:
    rgrMask --> matrix in which each subregion has a index number
'''

def fnShapeToMask(
    poly,                  # dictionary which contains polygons
    rgrlon,                    # lon coordinates of the target grid
    rgrlat,                     # lat coordinate of the target grid
    rgrLonBND,             # lon grid cell corners
    rgrLatBND):            # lat grid cell corners

    import sys, traceback
    import dateutil.parser as dparser
    import string
    from pdb import set_trace as stop
    import numpy as np
    import shapefile
    from shapely.geometry import Polygon
    from shapely.geometry import Point

    rgrShapePolygon = Polygon(poly)

    rgiSize=rgrlon.shape
    rgiMask = np.empty((rgiSize[0],rgiSize[1],))
    rgiMask[:] = np.NAN

    # loop over grid cells
    for lo in range(rgiSize[0]):
        for la in range(rgiSize[1]):
            cell = Polygon([(rgrLonBND[lo,la],rgrLatBND[lo,la]),
                            (rgrLonBND[lo+1,la],rgrLatBND[lo+1,la]),
                            (rgrLonBND[lo+1,la+1],rgrLatBND[lo+1,la+1]),
                            (rgrLonBND[lo,la+1],rgrLatBND[lo,la+1])])
            if cell.intersection(rgrShapePolygon).area/cell.area >= 0.5:
                rgiMask[lo,la]=1

            # if rgrShapePolygon.contains(Point(rgrlon[lo,la], rgrlat[lo,la])) == True:
            #     rgiMask[lo,la]=1

    return rgiMask







#!/usr/bin/env python
'''
    File name: fnCellBoundaries
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 20.03.2015
    Date last modified: 20.03.2015

    ##############################################################
    Purpos:

    Estimate the cell boundaries from the cell location of regular grids

    returns: rgrLonBND & rgrLatBND --> arrays of dimension [nlon,nlat]
    containing the cell boundaries of each gridcell in rgrlon and rgrlat
'''

def fnCellCorners(
    rgrLon,
    rgrLat):

    import numpy as np
    from pdb import set_trace as stop

    rgiSize = np.array(rgrLon).shape
    rgrLonBND=np.empty((rgiSize[0]+1,rgiSize[1]+1,))
    rgrLonBND[:] = np.NAN
    rgrLatBND=np.empty((rgiSize[0]+1,rgiSize[1]+1,))
    rgrLatBND[:] = np.NAN

    for lo  in range(rgiSize[0]+1):
        for la  in range(rgiSize[1]+1):
            if lo < rgiSize[0]-1 and la < rgiSize[1]-1:
                # All points except at the boundaries
                rgrLonBND[lo, la] = rgrLon[lo, la]-\
                    (rgrLon[lo+1, la+1]-rgrLon[lo, la])/2
                rgrLatBND[lo, la] = rgrLat[lo, la]-\
                    (rgrLat[lo+1, la+1]-rgrLat[lo, la])/2
            elif lo >= rgiSize[0]-1 and la < rgiSize[1]-1:
                # reight boundary second last row
                rgrLonBND[lo, la] = rgrLon[lo-1, la]+\
                    (rgrLon[lo-1, la]-rgrLon[lo-2, la+1])/2
                rgrLatBND[lo, la] = rgrLat[lo-1, la]-\
                    (rgrLat[lo-2, la+1]-rgrLat[lo-1, la])/2
            elif lo < rgiSize[0]-1 and la >= rgiSize[1]-1:
                # upper boundary second last row
                rgrLonBND[lo, la] = rgrLon[lo, la-1]-\
                    (rgrLon[lo+1, la-2]-rgrLon[lo, la-1])/2
                rgrLatBND[lo, la] = rgrLat[lo, la-1]-\
                    (rgrLat[lo+1, la-2]-rgrLat[lo, la-1])/2
            elif lo >= rgiSize[0]-1 and la >= rgiSize[1]-1:
                # upper right grid cells
                rgrLonBND[lo, la] = rgrLon[lo-1, la-1]-\
                    (rgrLon[lo-2, la-2]-rgrLon[lo-1, la-1])/2
                rgrLatBND[lo, la] = rgrLat[lo-1, la-1]-\
                    (rgrLat[lo-2, la-2]-rgrLat[lo-1, la-1])/2

    return rgrLonBND,rgrLatBND






#!/usr/bin/env python
'''
    File name: fnGetTarOro
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 26.03.2015
    Date last modified: 26.03.2015

    ##############################################################
    Purpos:

    reads in an digital elevation file and remapps it to the target grid

    returns: rgrTarORO --> DEM on the target grid
'''

from netCDF4 import Dataset
import numpy as np
from pdb import set_trace as stop
import pandas as pd

def fnGetTarOro(
    sGridFile,
    sLon,
    sLat,
    sHeight,
    rgrTarDat,
    rgrLonTar,
    rgrLatTar,
    sRemapMeth='remapcon'):

    from IO_funktions import fn_read_netcdf_grid
    rgrSubRegion=None
    rgrlon,rgrlat,rgrHeight,dcDomain=fn_read_netcdf_grid(sGridFile,
                                                             sLon,
                                                             sLat,
                                                             rgrSubRegion,
                                                             sHeightName=sHeight)
    rgrHeight=rgrHeight[None,:]
    rgrHeight=np.append(rgrHeight,rgrHeight,axis=0)
    from HelperFunctions import fnCDOregrid
    rgrHeight=pd.Panel(rgrHeight)
    dfTarORO=fnCDOregrid(rgrHeight,
                         rgrlon,
                         rgrlat,
                         rgrTarDat.ix[0:2,:],
                         rgrLonTar,
                         rgrLatTar,
                         sRemapMeth=sRemapMeth)

    rgrTarORO=np.array(dfTarORO)[0,:]

    return rgrTarORO






#!/usr/bin/env python
'''
    File name: fnGridToPoint
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 02.11.2015
    Date last modified: 02.11.2015

    ##############################################################
    Purpos:

    Derives a time series from a gridded dataset by inver distance
    averaging the four nearest gridpoints.

    The data set has to be [time, lon, lat]

    returns: rgrTimeSeries
'''

import datetime
import glob
import netCDF4
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import re
import pandas as pd
import calendar
import datetime
import re
pattern = re.compile(r'\W+')
import matplotlib.pyplot as plt
import sys
import subprocess


def fnGridToPoint(
    rgrData,
    rgrLon,
    rgrLat,
    rLon,
    rLat):


    # find the grid points which are closest to the station locations
    rgrDistance=np.zeros((4)); rgrDistance[:]=np.nan

    distance = (rgrLat-rLat)**2 + (rgrLon-rLon)**2
    distance1D = distance.ravel()
    distanceSORT=distance1D.argsort()[:4]
    rgrMinDist=distance1D[distanceSORT]
    rgrDistance[:]=rgrMinDist
    rgiMinDist=np.zeros((4,2)); rgiMinDist[:]=np.nan
    di=0
    while di < 4:
        try:
            rgiMinDist[di]=np.where(distance == rgrMinDist[di])
        except:
            # there is more than one point with the same distance
            for eq in range(len(np.where(distance == rgrMinDist[di])[0])):
                rgiMinDist[di+eq]=[np.where(distance == rgrMinDist[di])[0][eq],np.where(distance == rgrMinDist[di])[1][eq]]
            di=di+len(np.where(distance == rgrMinDist[di])[0])-1
        di=di+1
    rgiSize=rgrData.shape
    rgrStationTMP=np.zeros((rgiSize[0],4)); rgrStationTMP[:]=np.nan
    for gc in range(4):
        rgrStationTMP[:,gc] = rgrData[:,rgiMinDist[gc,0],rgiMinDist[gc,1]]
    # inverse distance averaging
    rgrTimeSeries=np.sum(rgrStationTMP[:,:]/rgrDistance[:],axis=1)/np.sum(1./rgrDistance[:])

    return rgrTimeSeries










'''
    File name: RelColIntegrHumid
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 10.04.2016
    Date last modified: 10.04.2016

    ##############################################################
    Purpos:

    This function returns the relative column integrated relative humidity

    # potential water vapor from Varmaghani 2012: An Analytical Formula for
    # Potential Water Vapor in an Atmosphere of Constant Lapse Rate

    Input:
    precipitable water in mm
    surface temperature in Celsius
    laps rate (optional) --> default = 4 K/km

    returns:
    IntegRelHum --> integrated comumn relative humidity
'''


def RelColIntegrHumid(PW,
             T2M,
             alpha=4.0):

    # defineed constants
    A=6.65851*10**(-10)
    B=3.639415922*10**(-8)
    C=3.44569192613*10**(-5)
    D=-7.0714633*10**(-3)
    E=3.342085312
    F=-1798.139526
    G=491541.7167

    T0=T2M
    z=10.34  # height of column integrated over
    R=287.
    TZ=T0-alpha*z
    rTranReg=(-1/(R*alpha))*(A*(TZ**6-T0**6)+B*(TZ**5-T0**5)+C*(TZ**4-T0**4)+D*(TZ**3-T0**3)+E*(TZ**2-T0**2)+F*(TZ-T0)+G*np.log((TZ+273.15)/(T0+273.15)))
    PWmax=rTranReg*1000

    IntegRelHum=(PW/PWmax)*100

    return IntegRelHum



'''
    File name: MaximumPrecWater
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 20.04.2016
    Date last modified: 20.04.2016

    ##############################################################
    Purpos:

    This function returns the maximum precipitable water for a given

    T2M   -- surface temperature [C]
    alpha -- laps rate
    z0      -- altitude of station [km]; default is 0

    # potential water vapor from Varmaghani 2012: An Analytical Formula for
    # Potential Water Vapor in an Atmosphere of Constant Lapse Rate

    Input:
    precipitable water in mm
    surface temperature in Celsius
    laps rate (optional) --> default = 4 K/km

    returns:
    IntegRelHum --> integrated comumn relative humidity
'''


def MaximumPrecWater(T2M,
                     alpha=4.0,
                     z0=None):

    # defineed constants
    A=6.65851*10**(-10)
    B=3.639415922*10**(-8)
    C=3.44569192613*10**(-5)
    D=-7.0714633*10**(-3)
    E=3.342085312
    F=-1798.139526
    G=491541.7167
    R=287.

    T0=T2M+(alpha*z0)
    z=11.0  # height of column integrated over
    if z0 == None:
        z0=0.

    # calculate PW for T0
    TZ=T0-alpha*z
    rTranReg=(-1/(R*alpha))*(A*(TZ**6-T0**6)+B*(TZ**5-T0**5)+C*(TZ**4-T0**4)+D*(TZ**3-T0**3)+E*(TZ**2-T0**2)+F*(TZ-T0)+G*np.log((TZ+273.15)/(T0+273.15)))
    PWmaxT0=rTranReg*1000

    # calculate PW between z=0 and z0
    TZ=T2M
    rTranReg=(-1/(R*alpha))*(A*(TZ**6-T0**6)+B*(TZ**5-T0**5)+C*(TZ**4-T0**4)+D*(TZ**3-T0**3)+E*(TZ**2-T0**2)+F*(TZ-T0)+G*np.log((TZ+273.15)/(T0+273.15)))
    PWmaxT1=rTranReg*1000

    PWmax=PWmaxT0- PWmaxT1

    return PWmax







'''
    File name: fnRemapConOperator
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 26.05.2017
    Date last modified: 26.05.2017

    ##############################################################
    Purpos:

    Generates an opperator to coservatively remapp data from a source
    rectangular grid to an target rectangular grid.

    returns: grConRemapOp --> opperator that contains the grid cells and
    their wheights of the source grid for each target grid cell
'''

def fnRemapConOperator(rgrLonS,rgrLatS,  # source grid
                       rgrLonT,rgrLatT):                  # trarget grid
    import numpy as np
    from pdb import set_trace as stop
    from shapely.geometry import Polygon, shape

    # check if the grids are given in 2D
    if len(rgrLonS.shape) == 1:
        rgrLonS1=np.asarray(([rgrLonS,]*rgrLatS.shape[0]))
        rgrLatS=np.asarray(([rgrLatS,]*rgrLonS.shape[0])).transpose()
        rgrLonS=rgrLonS1
    if len(rgrLonT.shape) == 1:
        rgrLonT1=np.asarray(([rgrLonT,]*rgrLatT.shape[0]))
        rgrLatT=np.asarray(([rgrLatT,]*rgrLonT.shape[0])).transpose()
        rgrLonT=rgrLonT1

    # All lon grids have to go from -180 to +180 --> convert now!'
    if np.min(rgrLonS) > 180:
        rgi180=np.where(rgrLonS > 180)
        rgrLonS[rgi180]=rgrLonS[rgi180]-360.
    if np.min(rgrLonT) > 180:
        rgi180=np.where(rgrLonT > 180)
        rgrLonT[rgi180]=rgrLonT[rgi180]-360.


    # get boundarie estimations for the grid cells since the center points
    # are given
    from HelperFunctions import fnCellCorners
    rgrLonSB,rgrLatSB=fnCellCorners(rgrLonS,rgrLatS)
    rgrLonTB,rgrLatTB=fnCellCorners(rgrLonT,rgrLatT)

    # get approximate grid spacing of source and target grid
    rGsS=(abs(np.mean(rgrLonS[:,1:]-rgrLonS[:,0:-1]))+abs(np.mean(rgrLatS[1:,:]-rgrLatS[0:-1,:])))/2
    rGsT=(abs(np.mean(rgrLonT[:,1:]-rgrLonT[:,0:-1]))+abs(np.mean(rgrLatT[1:,:]-rgrLatT[0:-1,:])))/2
    rRadius=((rGsS+rGsT)*1.2)/2.

    # loop over the target grid cells and calculate the weights
    grRemapOperator={}
    for la in range(rgrLonT.shape[0]):
        for lo in range(rgrLonT.shape[1]):
            rgbGCact=((rgrLonS > rgrLonT[la,lo]-rRadius) & (rgrLonS < rgrLonT[la,lo]+rRadius) & \
                      (rgrLatS > rgrLatT[la,lo]-rRadius) & (rgrLatS < rgrLatT[la,lo]+rRadius))
            if np.sum(rgbGCact) > 0:
                rgrLaLoArea=np.array([])
                # produce polygon for target grid cell
                pT=Polygon([(rgrLonTB[la,lo],rgrLatTB[la,lo]),(rgrLonTB[la,lo+1],rgrLatTB[la,lo+1]),(rgrLonTB[la+1,lo+1],rgrLatTB[la+1,lo+1]),(rgrLonTB[la+1,lo],rgrLatTB[la+1,lo])])
                # loop over source grid cells
                rgiGCact=np.where(rgbGCact == True)
                for sg in range(np.sum(rgbGCact)):
                    laS=rgiGCact[0][sg]
                    loS=rgiGCact[1][sg]
                    pS=Polygon([(rgrLonSB[laS,loS],rgrLatSB[laS,loS]),(rgrLonSB[laS,loS+1],rgrLatSB[laS,loS+1]),(rgrLonSB[laS+1,loS+1],rgrLatSB[laS+1,loS+1]),(rgrLonSB[laS+1,loS],rgrLatSB[laS+1,loS])])
                    rIntArea=pS.intersection(pT).area/pS.area
                    if rIntArea !=0:
                        if len(rgrLaLoArea) == 0:
                            rgrLaLoArea=np.array([[laS,loS,rIntArea]])
                        else:
                            rgrLaLoArea=np.append(rgrLaLoArea,np.array([[laS,loS,rIntArea]]), axis=0)
                grRemapOperator[str(la)+','+str(lo)]=rgrLaLoArea
    return grRemapOperator




'''
    File name: fnRemapCon
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 13.06.2017
    Date last modified: 13.06.2017

    ##############################################################
    Purpos:

    Uses the remap operator generated by the function fnRemapConOperator to
    remap data to a target grid conservatively

    returns: rgrTarData --> remapped data matrix
'''

def fnRemapCon(rgrLonS,rgrLatS,  # source grid
               rgrLonT,rgrLatT,                  # trarget grid
               grOperator,
               rgrData):
    import numpy as np
    from pdb import set_trace as stop

    if len(rgrData.shape) == 3:
        if len(rgrLonT.shape) == 1:
            rgrTarData=np.zeros((rgrData.shape[0], rgrLatT.shape[0],rgrLonT.shape[0])); rgrTarData[:]=np.nan
        elif len(rgrLonT.shape) == 2:
            rgrTarData=np.zeros((rgrData.shape[0], rgrLatT.shape[0],rgrLatT.shape[1])); rgrTarData[:]=np.nan
        for gc in range(len(grOperator.keys())):
            rgiGcT=np.array(grOperator.keys()[gc].split(',')).astype('int')
            rgrSource=grOperator[grOperator.keys()[gc]]
            if len(rgrSource) !=0:
                try:
                    rgrTarData[:,rgiGcT[0],rgiGcT[1]]=np.average(rgrData[:,rgrSource[:,0].astype('int'),rgrSource[:,1].astype('int')], weights=rgrSource[:,2], axis=1)
                except:
                    stop()
    if len(rgrData.shape) == 4:
        # the data has to be in [time, variables, lat, lon]
        rgrTarData=np.zeros((rgrData.shape[0], rgrData.shape[1], rgrLatT.shape[0],rgrLonT.shape[0])); rgrTarData[:]=np.nan
        for gc in range(len(grOperator.keys())):
            rgiGcT=np.array(grOperator.keys()[gc].split(',')).astype('int')
            rgrSource=grOperator[grOperator.keys()[gc]]
            rgrTarData[:,:,rgiGcT[0],rgiGcT[1]]=np.average(rgrData[:,:,rgrSource[:,0].astype('int'),rgrSource[:,1].astype('int')], weights=rgrSource[:,2], axis=2)

    return rgrTarData
