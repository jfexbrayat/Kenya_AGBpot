"""
27/07/2018 - JFE
this script plots a maps identifying the best 10% for forest growth in Kenya
that is not currently under crops
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from osgeo import gdal
import sys
import pandas as pd

path = '/disk/scratch/local.2/jexbraya/kenya_ODA/'

nc_med = Dataset(path+'/output/Kenya_ODA_v2_PFB_mean_WC2_SOTWIS_GridSearch.nc')

# load observed and potential AGB
obs = nc_med.variables['AGB_mean'][:]
pot = nc_med.variables['AGBpot_mean'][:]

#load land cover data
lcfile = gdal.Open(path+'/processed/ESACCI-LC-L4-LCCS-Map-1992-2015_30s.tif')
landcover = lcfile.GetRasterBand(24).ReadAsArray()
landcover = np.ma.array(landcover)
crops = (landcover>=10) & (landcover <=40)

#load legend just to have it available
lclegend= pd.read_csv(path+'/ESACCI-LC-Legend.csv',delimiter=';')

#get highest decile regardless of land cover
absolute_best10 = pot>=np.percentile(pot[~pot.mask],[90])[0]

#get highest decile if current crops are excluded
#first exclude currently cropped regions
dummy = pot.copy()
dummy[crops]=0.
realistic_best10 = dummy>=np.percentile(dummy[~dummy.mask],[90])[0]
realistic_best10.mask = absolute_best10.mask


#set extent and instantiate mappable items
ext = [33.5,42.5,-5,6]

brd= cfeat.BORDERS;brd.scale='10m'
lk = cfeat.LAKES;lk.scale='10m'
co = cfeat.COASTLINE; co.scale = '110m'
oc = cfeat.OCEAN; co.scale = '110m'
la = cfeat.LAND;la.scale='110m'

titles = ['a) Absolute best 10%','b) Best 10% of non currently cropped']
#create map and plot using axesgrid to nicely align map and colorbar
figmaps = plt.figure('maps',figsize=(12,8));figmaps.clf()

for ii,best10 in enumerate((absolute_best10,realistic_best10)):

    #assign to variable ax for simplicity
    ax = figmaps.add_subplot(1,2,ii+1,projection=ccrs.PlateCarree())
    im = ax.imshow(best10,origin='upper',vmin = 0, vmax=1,extent=ext,interpolation='nearest',cmap='viridis')

    #add features: ocean and land to have map on a light grey background
    ax.add_feature(oc,facecolor='silver',zorder=-1)
    ax.add_feature(la,facecolor='silver',zorder = -1)

    #set the lat/lon
    ax.set_xticks(np.arange(34,42.1,2))
    ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.set_yticks(np.arange(-4,5.1,2))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    #create proxy for legend
    proxy = [plt.Rectangle((0,0),1,1,fc=im.get_cmap()(ii)) for ii in np.arange(2.)[::-1]]

    ax.legend(proxy,('Optimal regions','Other'),loc='lower left')

    ax.set_title(titles[ii])
    #get potential biomass in these best 10%
    tot = ((pot*best10)*nc_med.variables['areas'][:]).sum()*1e-13*0.48
    print(ii,tot)


#figmaps.show()
figmaps.savefig('figures/identify_V2_WC2_SOTWIS.png', bbox_inches='tight')
