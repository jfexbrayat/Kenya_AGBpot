"""
27/07/2018 - JFE
this script plots maps of potential biomass in Kenya


"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
from osgeo import gdal

import sys

path = '/disk/scratch/local.2/jexbraya/kenya_ODA/'

nc_med = Dataset(path+'/output/Kenya_ODA_v31_AGBpot_mean_WC2_SOTWIS_GridSearch.nc')

# load observed and potential AGB
obs = nc_med.variables['AGB_mean'][:]
pot = nc_med.variables['AGBpot_mean'][:]

# JFE added forest mask to only plot forests
frst = nc_med.variables['training'][:] == 2
#replace places outside forests in obs as 0
#obs[~frst] = 0.

#set extent and instantiate mappable items
ext = [33.5,42.5,-5,6]

brd= cfeat.BORDERS;brd.scale='10m'
lk = cfeat.LAKES;lk.scale='10m'
co = cfeat.COASTLINE; co.scale = '110m'
oc = cfeat.OCEAN; co.scale = '110m'
la = cfeat.LAND;la.scale='110m'


#define projection
prj=ccrs.PlateCarree()

#create map and plot using axesgrid
ax = plt.subplot(111,projection = prj)

im = ax.imshow(nc_med.variables['training'][:],origin='upper',extent=ext,interpolation='nearest')
ax.add_feature(oc,facecolor='silver',zorder=-1)
ax.add_feature(la,facecolor='silver',zorder = -1)

#create proxy for legend
proxy = [plt.Rectangle((0,0),1,1,fc=im.get_cmap()(ii)) for ii in np.linspace(0,1,3)[::-1]]
ax.legend(proxy,('Forests','Bare areas'),loc='lower left')

#set the lat/lon
ax.set_xticks(np.arange(34,42.1,2))
ax.xaxis.set_major_formatter(LongitudeFormatter())

ax.set_yticks(np.arange(-4,5.1,2))
ax.yaxis.set_major_formatter(LatitudeFormatter())


plt.show()
plt.savefig('figures/training_regions.png', bbox_inches='tight',dpi=300)
