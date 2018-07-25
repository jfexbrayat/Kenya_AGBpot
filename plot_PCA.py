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

#load soil properties and get a mask
sotwisfiles = glob.glob(path+'/KEN_SOTWIS/*tif');sotwisfiles.sort()
soilmask = gdal.Open(sotwisfiles[0]).ReadAsArray()!=-9999.

#define coordinates
y,x = wc2mask.shape
lon = np.arange(x)*geo[1]+geo[0]+geo[1]/2.
lat = np.arange(y)*geo[-1]+geo[3]+geo[-1]/2.

#select prediction zone
slcpred = kenya*wc2mask*(lc2015!=210)*soilmask
print '# total pixels', slcpred.sum()

predfiles = wc2subset+sotwisfiles
predict = np.zeros([slcpred.sum(),len(predfiles)])
for vv,varfile in enumerate(predfiles):
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


xlabs = []
for i in range(1,20):
    xlabs.append('BI%02i' % i)
for fname in sotwisfiles:
    xlabs.append(fname.split('/')[-1].split('.')[0])
# plot \o/
fig = plt.figure('PCA',figsize=(8,5));fig.clf()
#add the heatmap at in an AxesGrid to make the colorbar and plots fit together
axgr=AxesGrid(fig,111,nrows_ncols=(1,1),label_mode='each',cbar_mode='single', \
cbar_pad = 0.05,cbar_size="3%",axes_pad=.55)

im=axgr[0].imshow(corrmat.T,vmin=-1,vmax=1,cmap='RdYlBu_r')
cb=axgr.cbar_axes[0].colorbar(im,ticks=[-1,-.5,0,.5,1])

axgr[0].set_xticks(np.arange(len(xlabs)))
axgr[0].set_xticklabels(xlabs,rotation=90)
axgr[0].set_xlabel('Predictor')
axgr[0].set_yticks(np.arange(corrmat.shape[1]))
axgr[0].set_yticklabels(np.arange(1,corrmat.shape[1]+1).astype('S'))
axgr[0].set_ylabel('Principal component')
cb.ax.set_title("Pearson's correlation", size = 'medium')

#show / save
#fig.show()
fig.savefig('figures/pca_results.png',bbox_inches='tight')
