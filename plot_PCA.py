'''
25/07/2018 - JFE
This script performs a PCA on the WorldClim2 bioclimatic indicators and plots
the following figures:
- a heatmap showing the correlation between the bioclim indicators and each of the
PC which cumulatively explain 95% of the variance of the data
- maps of the selected EOF values over Kenya
'''

# imports, not all used here but good to have in the interactive session
import glob
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import sys
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

#plotting / mapping libraries
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#define path
path = '/disk/scratch/local.2/jexbraya/kenya_ODA/processed/'

#get Kenya mask
kenya = gdal.Open(path+'/Kenya_AGB_2015_v2_sleek_mask_30arcsec.tif').ReadAsArray()!=65535

#get the forest fraction within each 30s cell
fraction = gdal.Open(path+'Kenya_sleek_mask_forest_fraction_30s.tif').ReadAsArray()
fraction[fraction==65535] = -9999.

#open file with agb and get GetGeoTransform
agbfile = gdal.Open(path+'/Kenya_AGB_2015_v2_sleek_mask_changed_nodata_30s.tif')
geo = agbfile.GetGeoTransform()
agbdata = agbfile.ReadAsArray()

#open landcover file and extract regions which are not water bodies
lcfile  = gdal.Open(path+'/ESACCI-LC-L4-LCCS-Map-1992-2015_30s.tif')
lc2015  = lcfile.GetRasterBand(24).ReadAsArray()

#get climate data and a mask
wc2files = glob.glob(path+'/wc2_hist/wc2*tif');wc2files.sort()
wc2subset = []
wc2vars = []
for ff,fname in enumerate(wc2files):
    variable = fname.split('_')[-2]
    if int(variable) in range(1,20):
        wc2vars.append(variable)
        wc2subset.append(fname)
        if variable == '01':
            dummy = gdal.Open(fname).ReadAsArray()
            wc2mask = np.ones(dummy.shape,dtype='bool')
            wc2mask[dummy < -1.69e308] = False

#define coordinates
y,x = wc2mask.shape
lon = np.arange(x)*geo[1]+geo[0]+geo[1]/2.
lat = np.arange(y)*geo[-1]+geo[3]+geo[-1]/2.

slcpred = kenya*wc2mask*(lc2015!=210)
print '# total pixels', slcpred.sum()

predict = np.zeros([slcpred.sum(),len(wc2subset)])
for vv,varfile in enumerate(wc2subset):
    predict[:,vv] = gdal.Open(varfile).ReadAsArray()[slcpred]

#create a pipeline to standardize and extract EOFs
pipeline = make_pipeline(StandardScaler(),PCA(n_components=0.95))
pipeline.fit(predict)

X_pred = pipeline.transform(predict)

#calculate a correlation matrix
corrmat = np.zeros([predict.shape[1],X_pred.shape[1]])
for ii in range(corrmat.shape[0]):
    for jj in range(corrmat.shape[1]):
        corrmat[ii,jj] = pearsonr(predict[:,ii],X_pred[:,jj])[0]

# plot \o/
fig = plt.figure('PCA',figsize=(8,5));fig.clf()

# start with the maps at top
prj=ccrs.PlateCarree()
axes_class = (GeoAxes,dict(map_projection=prj))

grmap = AxesGrid(fig,211,nrows_ncols=(1,5),axes_class=axes_class,label_mode='each',cbar_mode='single', \
cbar_pad = 0.05,cbar_size="15%",axes_pad=.05)
for ii in range(5):
    grmap[ii].set_facecolor('silver')
    dummy = np.zeros(slcpred.shape)-9999
    dummy[slcpred] = scale(X_pred)[:,ii]
    im = grmap[ii].imshow(np.ma.masked_equal(dummy,-9999.),origin='upper', \
    extent=[lon.min(),lon.max(),lat.min(),lat.max()],vmin=-2,vmax=2,cmap='RdYlBu_r')
    grmap[ii].set_title(chr(ord('a')+ii)+') PC%i: %.1f%%' % (ii+1,pipeline.steps[1][1].explained_variance_ratio_[ii]*100),fontsize='small')
    grmap[ii].add_feature(cfeat.LAND,facecolor='silver',zorder=-1)
    grmap[ii].add_feature(cfeat.OCEAN,facecolor='silver',zorder=-1)

cb= grmap.cbar_axes[0].colorbar(im,extend='both',ticks=[-2,-1,0,1,2],drawedges=False)
#cb.set_label_text("Standardized PC values")

#add the heatmap at the bottom using AxesGrid to make the colorbar and plots
#fit together
axgr=AxesGrid(fig,212,nrows_ncols=(1,1),label_mode='each',cbar_mode='single', \
cbar_pad = 0.05,cbar_size="3%",axes_pad=.55)

im=axgr[0].imshow(corrmat.T,vmin=-1,vmax=1,cmap='RdYlBu_r')
cb=axgr.cbar_axes[0].colorbar(im,ticks=[-1,-.5,0,.5,1])


axgr[0].set_xticks(np.arange(19));axgr[0].set_xticklabels(np.arange(1,20).astype('S'))
axgr[0].set_xlabel('Bioclimatic indicator')
axgr[0].set_yticks(np.arange(5));axgr[0].set_yticklabels(np.arange(1,6).astype('S'))
axgr[0].set_ylabel('Principal components')
axgr[0].set_title("f) Pearson's correlation")

#show / save
fig.show()
fig.savefig('pca_results.png',bbox_inches='tight')
