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
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
from osgeo import gdal

import sys

path = '/disk/scratch/local.2/jexbraya/kenya_ODA/'

nc_med = Dataset(path+'/output/Kenya_ODA_v31_AGBpot_mean_WC2_SOTWIS_GridSearch.nc')
xr_upp = xr.open_dataset(path+'/output/Kenya_ODA_v31_AGBpot_upper_WC2_SOTWIS_GridSearch.nc')
xr_low = xr.open_dataset(path+'/output/Kenya_ODA_v31_AGBpot_lower_WC2_SOTWIS_GridSearch.nc')

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

titles = ['a) AGB','b) Potential Forest Biomass (PFB)','c) PFB-AGB']

#define projection
prj=ccrs.PlateCarree()
axes_class = (GeoAxes,dict(map_projection=prj))

#create map and plot using axesgrid
figmaps = plt.figure('maps',figsize=(12,8));figmaps.clf()

axgr = AxesGrid(figmaps,111,nrows_ncols=(1,3),axes_class=axes_class,label_mode='',cbar_mode='each',cbar_pad = 0.04,cbar_size="5%",axes_pad=1.)

vmn = [0,0,0]
vmx = [200,200,100]
cmaps = ['viridis','viridis','plasma']

#obs[(nc_med.variables['training'][:]==1)] = 0.
#obs.data[(obs.mask)*(~pot.mask)] = 0.
#obs.mask[(obs.mask)*(~pot.mask)] = False

titles = ['a) AGB','b) AGB$_{pot}$','c) AGB$_{pot}$ - AGB']

#iterate maps on the grid
for mm,map2plot in enumerate([obs,pot,pot-obs]):

    #plot
    ax = axgr[mm]
    #if mm == 2:
    #    map2plot.mask[nc_med.variables['training'][:]>0] = True
    im = ax.imshow(map2plot,origin='upper',vmin = vmn[mm],vmax=vmx[mm],extent=ext,interpolation='nearest',cmap=cmaps[mm])
    #add colorbar and label on rightmost one
    cb = axgr.cbar_axes[mm].colorbar(im)
    if mm ==2:
        cb.set_label_text('Mg ha$^{-1}$')
    #add features: ocean and land to have map on a light grey background
    ax.add_feature(oc,facecolor='silver',zorder=-1)
    ax.add_feature(la,facecolor='silver',zorder = -1)


    #ax.text(0.99,0.99,titles[mm],transform = ax.transAxes,va='top',ha='right',weight='bold',size='medium')
    ax.set_title(titles[mm])

    #set the lat/lon
    ax.set_xticks(np.arange(34,42.1,2))
    ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.set_yticks(np.arange(-4,5.1,2))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    print(mm, (map2plot*nc_med.variables['areas'][:]).sum()*1e-13)

print('Range of AGB: %4.2f Pg C - %4.2f Pg C' % ((xr_low.AGB_lower*xr_low.areas).sum()*1e-13*0.48,(xr_upp.AGB_upper*xr_upp.areas).sum()*1e-13*0.48))
print('Range of AGB: %4.2f Pg C - %4.2f Pg C' % ((xr_low.AGBpot_lower*xr_low.areas).sum()*1e-13*0.48,(xr_upp.AGBpot_upper*xr_upp.areas).sum()*1e-13*0.48))
#figmaps.show()
figmaps.savefig('figures/compare_maps_v31_WC2_SOTWIS.png', bbox_inches='tight', dpi=300)
