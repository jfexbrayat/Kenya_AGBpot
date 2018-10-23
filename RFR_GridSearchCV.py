'''
23/10/2018 - JFE
the version of the dataset to be used is now a command line argument

10/09/2018 - JFE
changed input map to be version 3.1 and adjusted land use codes from ESA-CCI to
include 90 as forest cover

24/07/2018 - JFE
Added a PCA to reduce dimensionality

26/5/2018 -JFE
updates include using ESACCI during 1992-2015 to:
- select areas which have been bare (code 200 to 202) as "desert attractor"
- select areas which have been forests (code 50 to 82) as training regions


'''

import matplotlib;matplotlib.use('Agg')
import glob
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

#get  climate data mask
wc2files = glob.glob(path+'/wc2_hist/wc2*tif');wc2files.sort()
wc2subset = []
wc2vars = []
for ff,fname in enumerate(wc2files):
    variable = fname.split('_')[-2]
    #if int(variable) in ['01','04','05','06','12','13','14','15']:
    if int(variable) in range(1,20):
        wc2vars.append(variable)
        wc2subset.append(fname)
        if variable == '01':
            dummy = gdal.Open(fname).ReadAsArray()
            wc2mask = np.ones(dummy.shape,dtype='bool')
            wc2mask[dummy < -1.69e308] = False

#get soil data mask
sotwisfiles = glob.glob(path+'/KEN_SOTWIS/*tif');sotwisfiles.sort()
#get variable names
sotwisvars = [f.split('/')[-1].split('.')[0] for f in sotwisfiles]
soilmask = gdal.Open(sotwisfiles[0]).ReadAsArray()!=-9999.

#defin coordinates
y,x = wc2mask.shape
lon = np.arange(x)*geo[1]+geo[0]+geo[1]/2.
lat = np.arange(y)*geo[-1]+geo[3]+geo[-1]/2.

#calculate area
areas = np.zeros([lat.size,lon.size])
res = np.abs(lat[1]-lat[0])
for la,latval in enumerate(lat):
    areas[la]= (6371e3)**2 * ( np.deg2rad(0+res/2.)-np.deg2rad(0-res/2.) ) * (np.sin(np.deg2rad(latval+res/2.))-np.sin(np.deg2rad(latval-res/2.)))
lon2d,lat2d = np.meshgrid(lon,lat)

#set target
target = np.zeros(kenya.shape)-9999.
target[frst] = agbdata[frst]
target[bare] = 0.
target[target==65535] = -9999.
target = np.ma.masked_equal(target,-9999.)

#replace in case uncertainty is greater than mean value
print('range of target data', target.min(), target.max())

slc = ~target.mask * wc2mask * soilmask
print('# of training pixels', slc.sum())

#adjust the frst and bare masks to match the actual selected data points
frst[~slc] = False
bare[~slc] = False
print('# of forest pixels: ', frst.sum())
print('# of bare pixels: ', bare.sum())

# extract data for final prediction here, needed for PCA
# selection is Kenya, wc2 mask excluding water bodies (code 210)
slcpred = kenya*wc2mask*(lc2015!=210)*soilmask
print('# of land pixels: ', slcpred.sum())

predfiles = wc2subset+sotwisfiles
predict = np.zeros([slcpred.sum(),len(predfiles)])
for vv,varfile in enumerate(predfiles):
    predict[:,vv] = gdal.Open(varfile).ReadAsArray()[slcpred]

#create a pipeline to standardize and extract EOFs
pipeline = make_pipeline(StandardScaler(),PCA(n_components=0.95))
pipeline.fit(predict)

X_pred = pipeline.transform(predict)
X = X_pred[slc[slcpred]]
#X = predict[slc[slcpred]]
#training = np.column_stack([training,pd.get_dummies(data['TAXNWRB'][slc]).get_values()])

lat_pixels = lat2d[slc]
lon_pixels = lon2d[slc]

y = target.data[slc]

if sys.argv[3] == 'new':

    print('New application')
    forest = RF(n_jobs = -1, oob_score=True,random_state = 26)

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=26)

    print(X_train.shape,X_test.shape)

    param_grid = {"max_features": ['auto','sqrt','log2'],
          "min_samples_leaf": np.arange(20,60,10),
          "n_estimators": [100,200,500,1000]}

    grid = GridSearchCV(forest,param_grid=param_grid,cv=3,verbose = True,\
    scoring = 'neg_mean_squared_error')

    grid.fit(X_train,y_train)

    print(grid.score(X_train,y_train),np.sqrt(mean_squared_error(y_train,grid.predict(X_train))))
    print(grid.score(X_test,y_test),np.sqrt(mean_squared_error(y_test,grid.predict(X_test))))

    forest = grid.best_estimator_

    fig = pl.figure(figsize=(10,8))
    for ii,calval in enumerate([(X_train,y_train),(X_test,y_test)]):
        ax = fig.add_subplot(1,2,ii+1)
        plot_OLS(ax,calval[1],forest.predict(calval[0]),mode='unicolor')
    fig.axes[0].set_title('Calibration')
    fig.axes[1].set_title('Validation')
    #fig.show()
    fig.savefig('calval/%s/calval_%s_%s_WC2_SOTWIS_GridSearch.png' % (version,version,lvl),bbox_inches='tight')

    #now fit final forest on all dataset
    forest.fit(X,y)

    #save the fitted algorithm to avoid refitting everytime
    joblib.dump(forest,path+'/../saved_algorithms/kenya_ODA_%s_AGBpot_%s_WC2_SOTWIS.pkl' % (version,lvl),compress = 1)

elif sys.argv[3] == 'load':

    print('Loading existing application')
    forest = joblib.load(path+'/../saved_algorithms/kenya_ODA_%s_AGBpot_%s_WC2_SOTWIS.pkl' % (version,lvl))

print(forest.score(X,y),np.sqrt(mean_squared_error(y,forest.predict(X))))
print("AGB in training data: %4.2f Pg" % ((target*areas).sum()*1e-13))

#create new map of potential forest biomass
#X_pred = predict
potmap = np.zeros(kenya.shape)-9999.
potmap[slcpred] = forest.predict(X_pred)
print("AGB in trained model: %4.2f Pg" % ((np.ma.masked_equal(potmap,-9999)*areas)[slc].sum()*1e-13))

potmap[slc] = y
potmap[~slcpred] = -9999.
potmap = np.ma.masked_equal(potmap,-9999.)

print("AGB in Pedro's map: %4.2f Pg C" % ((np.ma.masked_equal(agbdata,65535)*areas).sum()*1e-13*0.48))
print("Potential AGB     : %4.2f Pg C" % ((np.ma.masked_equal(potmap,-9999.)*areas).sum()*1e-13*0.48))

figimp = pl.figure('imp');figimp.clf()

ax= figimp.add_subplot(111)
#sort importances
idx = np.argsort(forest.feature_importances_)[::-1]
imp = forest.feature_importances_[idx]
impstd = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)[idx]
ax.bar(range(imp.size),imp,color='r',yerr=impstd,align='center')

ax.set_xticks(range(imp.size))
ax.set_xticklabels(np.array(wc2vars+sotwisvars)[idx],fontsize='small',rotation=90)
ax.set_xlabel('variable')

ax.set_ylabel('variable importance')
ax.set_ylim(0,ax.get_ylim()[1])
#figimp.show()
figimp.savefig('calval/%s/importances_%s_%s_WC2_SOTWIS_GridSearch.png' % (version,version,lvl),bbox_inches='tight')

#save a netcdf file if needed
if sys.argv[4] =='savenc':
    import os

    fname = 'Kenya_ODA_%s_AGBpot_%s_WC2_SOTWIS_GridSearch.nc' % (version,lvl)

    if fname in os.listdir(path+'/../output/'):
        os.remove(path+'/../output/'+fname)


    nc = Dataset(path+'/../output/'+fname,'w')

    nc.createDimension('lon',size=lon.size)
    nc.createDimension('lat',size=lat.size)

    nc.createVariable('lat','d',dimensions=('lat'))
    nc.variables['lat'][:] = lat
    nc.variables['lat'].units='degrees_north'

    nc.createVariable('lon','d',dimensions=('lon'))
    nc.variables['lon'][:] = lon
    nc.variables['lon'].units='degrees_east'

    agbmap = agbdata.astype('float')
    agbmap[agbmap==65535.] = -9999.

    nc.createVariable('AGB_%s' % lvl,'d',dimensions=('lat','lon'), zlib = True)
    nc.variables['AGB_%s' % lvl][:] = agbmap
    nc.variables['AGB_%s' % lvl].missing_value = -9999.
    nc.variables['AGB_%s' % lvl].long_name = 'Forest AGB_%s ODA map %s regridded to 30arcsec' % (version,lvl)
    nc.variables['AGB_%s' % lvl].units = 'Mg ha-1'

    nc.createVariable('AGBpot_%s' % lvl,'d',dimensions=('lat','lon'), zlib = True)
    nc.variables['AGBpot_%s' % lvl][:] = potmap.data
    nc.variables['AGBpot_%s' % lvl].missing_value = -9999.
    nc.variables['AGBpot_%s' % lvl].units = 'Mg ha-1'
    nc.variables['AGBpot_%s' % lvl].long_name = "AGBpot_%s constructed using Kenya's ODA forest AGB_%s map %s" % (lvl,lvl,version)

    #nc.createVariable('forestfraction','d',dimensions=('lat','lon'), zlib = True)
    #nc.variables['forestfraction'][:] = fraction
    #nc.variables['forestfraction'].missing_value = -9999.
    #nc.variables['forestfraction'].units = '-'
    #nc.variables['forestfraction'].long_name = 'fraction of pixel currently forested'

    nc.createVariable('training','d',dimensions=('lat','lon'), zlib = True)
    train = np.zeros(kenya.shape,dtype='i')-9999.
    train[kenya] = 0.
    train[bare] = 1.
    train[frst] = 2.
    nc.variables['training'][:] = train
    nc.variables['training'].long_name = 'regions used as training: 1: bare; 2: forests'
    nc.variables['training'].missing_value = -9999.

    nc.createVariable('areas','d',dimensions=('lat','lon'), zlib = True)
    nc.variables['areas'][:] = areas
    nc.variables['areas'].long_name = 'Pixel area'
    nc.variables['areas'].units = 'm2'

    nc.sync();nc.close()
