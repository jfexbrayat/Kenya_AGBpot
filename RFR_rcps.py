'''
31/10/2018 - JFE
this file produces AGBpot map according to different scenarios from wc1
'''

import matplotlib;matplotlib.use('Agg')
import glob,os
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.api as sm
import pylab as pl
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import sys
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import xarray as xr

def plot_OLS(ax,target,Y,mode='unicolor'):

    X = target
    X = sm.add_constant(X)

    model = sm.OLS(Y,X)

    results = model.fit()

    st, data, ss2 = summary_table(results, alpha=0.05)

    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    if mode == 'unicolor':
        ax.scatter(target,Y,c='silver',linewidths=0, s =4)
    else:
        xy = np.row_stack([target,Y])
        z = gaussian_kde(xy)(xy)
        idx=z.argsort()
        x,y,z = xy[0][idx],xy[1][idx],z[idx]
        ax.scatter(x,y,c=z,s=4,cmap=pl.cm.inferno_r)

    ax.plot(target,fittedvalues,'r-',label='Least Square Regression',lw=2)

    idx = np.argsort(predict_ci_low)
    ax.plot(target[idx],predict_ci_low[idx],'r--',lw=2,label='95% confidence interval')

    idx = np.argsort(predict_ci_upp)
    ax.plot(target[idx],predict_ci_upp[idx],'r--',lw=2)

    mx = np.ceil(max(target.max(),fittedvalues.max()))
    ax.plot([0,mx],[0,mx],'k-')

    ax.set_xlim(0,mx)
    ax.set_ylim(0,mx)

    ax.set_aspect(1)

    ax.legend(loc='upper left')
    ax.set_xlabel('AGB from map [Mg ha$^{-1}$]')

    ax.set_ylabel('Reconstructed AGB [Mg ha$^{-1}$]')

    nse = 1-((Y-target)**2).sum()/((target-target.mean())**2).sum()
    rmse = np.sqrt(((Y-target)**2).mean())

    ax.text(0.98,0.02,'y = %4.2fx + %4.2f\nR$^2$ = %4.2f; p < 0.001\nrmse = %4.1f Mg ha$^{-1}$ ; NSE = %4.2f' % (results.params[1],results.params[0],results.rsquared,rmse,nse),va='bottom',ha='right',transform=ax.transAxes)

    idx = np.argsort(predict_ci_upp)
    ax.plot(target[idx],predict_ci_upp[idx],'r--',lw=2)

    mx = np.ceil(max(target.max(),fittedvalues.max()))
    ax.plot([0,mx],[0,mx],'k-')

    ax.set_xlim(0,mx)
    ax.set_ylim(0,mx)

    ax.set_aspect(1)

    ax.legend(loc='upper left')
    ax.set_xlabel('AGB from map [Mg ha$^{-1}$]')

    ax.set_ylabel('Reconstructed AGB [Mg ha$^{-1}$]')

    nse = 1-((Y-target)**2).sum()/((target-target.mean())**2).sum()
    rmse = np.sqrt(((Y-target)**2).mean())

    ax.text(0.98,0.02,'y = %4.2fx + %4.2f\nR$^2$ = %4.2f; p < 0.001\nrmse = %4.1f Mg ha$^{-1}$ ; NSE = %4.2f' % (results.params[1],results.params[0],results.rsquared,rmse,nse),va='bottom',ha='right',transform=ax.transAxes)

path = '/disk/scratch/local.2/jexbraya/kenya_ODA/processed/'

version = sys.argv[1]

#get Kenya mask
kenya = gdal.Open(path+'/Kenya_AGB2015_%s_30s.tif' % version).ReadAsArray()!=65535 # JFE replaced source file 10/09/2018
#get the forest fraction within each 30s cell
#fraction = gdal.Open(path+'Kenya_sleek_mask_forest_fraction_30s.tif').ReadAsArray()
#fraction[fraction==65535] = -9999.
#open file with agb and get GetGeoTransform
agbfile = gdal.Open(path+'/Kenya_AGB2015_%s_30s.tif' % version)
geo = agbfile.GetGeoTransform()
agbdata = agbfile.ReadAsArray()

lvl = sys.argv[2]
rcp = sys.argv[3]

#added version

if lvl in ['upper','lower']:
    uc = gdal.Open(path+'/Kenya_RelSTD2015_%s_30s.tif' % version).ReadAsArray()*0.01
    if lvl == 'upper':
        print('setting target as mean + uc')
        agbdata = agbdata+uc*agbdata
    elif lvl == 'lower':
        print('setting target as mean - uc')
        agbdata = agbdata-uc*agbdata
elif lvl == 'mean':
    print('keep target')
else:
    lvl = 'mean'
    print('no uncertainty level recognised, assuming mean')
agbdata[agbdata==agbdata[0,0]] = 65535
agbdata[agbdata<0] = 0

#open landcover file and extract forest and bare for 1992 and 2015
lcfile  = gdal.Open(path+'/ESACCI-LC-L4-LCCS-Map-1992-2015_30s.tif')
lc1992  = lcfile.GetRasterBand(1).ReadAsArray()
bare1992= (lc1992>=200)*(lc1992<=202)
frst1992= (lc1992>=50)*(lc1992<=90) # included code 90: mixed tree as forest - JFE 10/09/18
lc2015  = lcfile.GetRasterBand(24).ReadAsArray()
bare2015= (lc2015>=200)*(lc2015<=202)
frst2015= (lc2015>=50)*(lc2015<=90)

#final masks
bare = bare1992*bare2015*kenya
frst = frst1992*frst2015*kenya

print("forests in 1992: ", (frst1992*kenya).sum()/kenya.sum() )
print("forests in 2015: ", (frst2015*kenya).sum()/kenya.sum() )
print("forest from 1992 to 2015", (frst*kenya).sum()/kenya.sum() )

#get soil data mask
sotwisfiles = glob.glob(path+'/KEN_SOTWIS/*tif');sotwisfiles.sort()
#get variable names
sotwisvars = [f.split('/')[-1].split('.')[0] for f in sotwisfiles]
soilmask = gdal.Open(sotwisfiles[0]).ReadAsArray()!=-9999.

#defin coordinates
y,x = soilmask.shape
lon = np.arange(x)*geo[1]+geo[0]+geo[1]/2.
lat = np.arange(y)*geo[-1]+geo[3]+geo[-1]/2.

#calculate area
areas = np.zeros([lat.size,lon.size])
res = np.abs(lat[1]-lat[0])
for la,latval in enumerate(lat):
    areas[la]= (6371e3)**2 * ( np.deg2rad(0+res/2.)-np.deg2rad(0-res/2.) ) * (np.sin(np.deg2rad(latval+res/2.))-np.sin(np.deg2rad(latval-res/2.)))
lon2d,lat2d = np.meshgrid(lon,lat)

#load the pipeline to standardize and extract EOFs
pipeline = joblib.load(path+'/../saved_algorithms/pca_pipeline.pkl')
#load the RF
forest = joblib.load(path+'/../saved_algorithms/kenya_ODA_%s_AGBpot_%s_WC2_SOTWIS.pkl' % (version,lvl))

#loop over model to get data
models = sorted(os.listdir(path+'wc1_%s_70' % rcp))
dummy = np.zeros([len(models)]+list(lon2d.shape))-9999.

attrs = {'_FillValue':-9999.,'units':'Mg ha-1'}
xr_agbpot = xr.Dataset(data_vars={'AGBpot': (['model','lat','lon'],dummy.copy(),attrs)},
                        coords={'model':(['model'],np.array(models)),
                                'lat':(['lat'],lat,{'units':'degrees_north'}),
                                'lon':(['lon'],lon,{'units':'degrees_east'})})


for mm,model in enumerate(models):
    print(model)
    wc1files = sorted(glob.glob(path+'wc1_%s_70/%s/*.tif' % (rcp,model)))
    predfiles = wc1files+sotwisfiles
    for vv,varfile in enumerate(predfiles):
        #define slcpred
        if mm == 0 & vv == 0:
            wc1mask = gdal.Open(varfile).ReadAsArray()!=-32768
            slcpred = kenya*wc1mask*(lc2015!=210)*soilmask
            predict = np.zeros([slcpred.sum(),len(predfiles)])
        #wc1 is save as integers and have to apply a 0.1 correction to temperatures
        if vv in [0,1,3,4,5,6,7,8,9,10]:
            predict[:,vv] = gdal.Open(varfile).ReadAsArray()[slcpred]*.1
        else:
            predict[:,vv] = gdal.Open(varfile).ReadAsArray()[slcpred]
    X_pred = pipeline.transform(predict)
    xr_agbpot.AGBpot.values[mm,slcpred] = forest.predict(X_pred)

if sys.argv[4] == 'savenc':
    xr_agbpot.to_netcdf(path+'../output/Kenya_ODA_v21_AGBpot_%s_SOTWIS_%s.nc' % (lvl,rcp))
