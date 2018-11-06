"""
06/11/2018 - JFE
plots the results as barplots

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
xr_med_rcp45 = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_mean_SOTWIS_rcp45.nc')
xr_med_rcp85 = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_mean_SOTWIS_rcp85.nc')

xr_low = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_lower_WC2_SOTWIS_GridSearch.nc')
xr_low_rcp45 = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_lower_SOTWIS_rcp45.nc')
xr_low_rcp85 = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_lower_SOTWIS_rcp85.nc')

xr_upp = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_upper_WC2_SOTWIS_GridSearch.nc')
xr_upp_rcp45 = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_upper_SOTWIS_rcp45.nc')
xr_upp_rcp85 = xr.open_dataset(path+'/output/Kenya_ODA_v21_AGBpot_upper_SOTWIS_rcp85.nc')

# load observed and potential AGB

pot = xr_med.AGBpot_mean
pot45=xr_med_rcp45.AGBpot.mean(axis=0)
pot85=xr_med_rcp85.AGBpot.mean(axis=0)
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
best10_thresh_nocrop45= sorted(pot45.values[notfrst_notcrop])[-needed]
best10_thresh_nocrop85= sorted(pot85.values[notfrst_notcrop])[-needed]
#create masks
best10 = (notfrst*(pot>best10_thresh))
best10_nocrop = (notfrst_notcrop*(pot>best10_thresh_nocrop))

# now calculates budgets
fig = plt.figure()
meds = [(xr_med.AGB_mean*frst2015*xr_med.areas).sum()*.48*1e-10]
meds.append((xr_med.AGBpot_mean*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10)
meds.append((xr_med_rcp45.AGBpot.median(axis=0)*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10)
meds.append((xr_med_rcp85.AGBpot.median(axis=0)*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10)

err = [[(xr_upp.AGB_upper*frst2015*xr_med.areas).sum()*.48*1e-10,(xr_low.AGB_lower*frst2015*xr_med.areas).sum()*.48*1e-10]]
err.extend([[(xr_upp.AGBpot_upper*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10,(xr_low.AGBpot_lower*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10]])
err.extend([[(xr_upp_rcp45.AGBpot.median(axis=0)*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10,(xr_low_rcp45.AGBpot.median(axis=0)*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10]])
err.extend([[(xr_upp_rcp85.AGBpot.median(axis=0)*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10,(xr_low_rcp85.AGBpot.median(axis=0)*(frst2015+best10_nocrop)*xr_med.areas).sum()*.48*1e-10]])

err = np.array(err)

plt.bar(range(len(meds)),meds)
plt.vlines(range(len(meds)),ymax=err[:,0],ymin=err[:,1])

fig.show()




