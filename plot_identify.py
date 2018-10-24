"""
27/07/2018 - JFE
this script plots a maps identifying the best 10% for forest growth in Kenya
that is not currently under crop
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
import xarray as xr

path = '/disk/scratch/local.2/jexbraya/kenya_ODA/'

xr_med = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_mean_WC2_SOTWIS_GridSearch.nc')
xr_low = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_lower_WC2_SOTWIS_GridSearch.nc')
xr_upp = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_upper_WC2_SOTWIS_GridSearch.nc')

# load observed and potential AGB
obs = xr_med.AGB_mean
pot = xr_med.AGBpot_mean
# get kenya from the variables
kenya = ~pot.isnull()

#load land cover data for 2015
lcfile = gdal.Open(path+'/processed/ESACCI-LC-L4-LCCS-Map-1992-2015_30s.tif')
landcover = lcfile.GetRasterBand(24).ReadAsArray()
landcover = np.ma.array(landcover)
crop2015 = (landcover>=10) & (landcover <= 40) & kenya
frst2015 = (landcover>=50) & (landcover <= 90) & kenya
notfrst = kenya & ~frst2015
notfrst_notcrop = kenya & ~frst2015 & ~crop2015
#load legend just to have it available
lclegend= pd.read_csv(path+'/ESACCI-LC-Legend.csv',delimiter=';')

#get the number of pixels needed to represent 10% of Kenya
ten_percent = int(kenya.sum()/10)
#and the n of pixels needed to reach 10% of forests
needed = (ten_percent-frst2015.sum()).values

#get AGBpot thresh for all places but current forests
best10_thresh = sorted(pot.values[notfrst])[-needed]
#get AGBpot for all places but current forests and crop
best10_thresh_nocrop   = sorted(pot.values[notfrst_notcrop])[-needed]
#create masks
best10 = (notfrst*(pot>best10_thresh))
best10_nocrop = (notfrst_notcrop*(pot>best10_thresh_nocrop))

#write some stats -
print('AGBpot in target areas: %.1f Tg C' % ((xr_med.areas*pot*best10).sum()*1e-10*0.48))
print('AGBpot range in target areas: %.1f Tg C - %.1f Tg C' %
        ((xr_med.areas*xr_low.AGBpot_lower*best10).sum()*1e-10*0.48,
        (xr_med.areas*xr_upp.AGBpot_upper*best10).sum()*1e-10*0.48))
print('Fraction of target areas which are crop: %4.1f%%' % (100*(best10 & crop2015).sum()/(best10.sum())))

print('AGBpot in target areas wo crop: %.1f Tg C' % ((xr_med.areas*pot*best10_nocrop).sum()*1e-10*0.48))
print('AGBpot range in target areas wo crop: %.1f Tg C - %.1f Tg C' %
        ((xr_med.areas*xr_low.AGBpot_lower*best10_nocrop).sum()*1e-10*0.48,
        (xr_med.areas*xr_upp.AGBpot_upper*best10_nocrop).sum()*1e-10*0.48))

#set extent and instantiate mappable items
ext = [33.5,42.5,-5,6]

brd= cfeat.BORDERS;brd.scale='10m'
lk = cfeat.LAKES;lk.scale='10m'
co = cfeat.COASTLINE; co.scale = '110m'
oc = cfeat.OCEAN; co.scale = '110m'
la = cfeat.LAND;la.scale='110m'

titles = ['a) Optimal target areas','b) Realistic target areas']
#create map and plot using axesgrid to nicely align map and colorbar
figmaps = plt.figure('maps identify',figsize=(12,8));figmaps.clf()

for ii,best in enumerate((best10,best10_nocrop)):

    map2plot = best.copy().astype('int')*2.
    map2plot.values[frst2015] = 1
    map2plot.values[~kenya]=np.nan
    #assign to variable ax for simplicity
    ax = figmaps.add_subplot(1,2,ii+1,projection=ccrs.PlateCarree())
    im = ax.imshow(map2plot,origin='upper',vmin = 0, vmax=2,extent=ext,interpolation='nearest',cmap='viridis')

    #add features: ocean and land to have map on a light grey background
    ax.add_feature(oc,facecolor='silver',zorder=-1)
    ax.add_feature(la,facecolor='silver',zorder = -1)

    #set the lat/lon
    ax.set_xticks(np.arange(34,42.1,2))
    ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.set_yticks(np.arange(-4,5.1,2))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    #create proxy for legend
    proxy = [plt.Rectangle((0,0),1,1,fc=im.get_cmap()(ii)) for ii in np.linspace(0,1,3)[::-1]]

    ax.legend(proxy,('Target areas','Current forests','Other'),loc='lower left')

    ax.set_title(titles[ii],size='x-large')
    #ax.text(0.015,0.985,titles[ii],transform=ax.transAxes,va='top',fontsize='large')

    #get potential biomass in these best 10%
    tot = ((pot*best)*xr_med.areas).sum()*1e-13*0.48
    print(ii,tot)


#figmaps.show()
figmaps.savefig('figures/identify_v21_WC2_SOTWIS.png', bbox_inches='tight',dpi=300)
