

# =============================================================================
# region import packages


# basic library
import pywt.data
import pywt
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc
from scipy import stats
import tables as tb

import sys  # print(sys.path)
sys.path.append(
    '/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
sys.path.append('/scratch/snx3000/qgao')


# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.animation as animation

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
# mpl.rcParams['backend'] = 'Qt4Agg'  #
# mpl.get_backend()

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"
from geopy import distance

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
    framework_plot1,
)


# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py
from scipy.ndimage import median_filter

from DEoAI_analysis.module.vortex_namelist import (
    correctly_identified
)

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
    timing,
    quantiles,
    folder_1km,
    g,
    m,
    r0,
    cp,
    r,
    r_v,
    p0sl,
    t0sl,
    extent1km,
    extent3d_m,
    extent3d_g,
    extent3d_t,
    extentm,
    extentc,
    extent12km,
    extent1km_lb,
    extent_global,
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
    ticklabel_global,
    transform,
    coastline,
    borders,
    rvor_level,
    rvor_ticks,
    rvor_cmp,
    center_madeira,
    angle_deg_madeira,
    radius_madeira,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    find_nearest_grid,
    rotate_wind,
    sig_coeffs,
    vortex_identification,
    vortex_identification1,
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    inversion_layer,
)


# endregion
# =============================================================================


# =============================================================================
# region plot vertical profile at grid nearest to funchal and radiosonde

#### Find nearest grid
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist[0:2],
    data_vars='minimal', coords='minimal', compat='override')
funchal_loc = np.array([32.6333, -16.9000])
nearestgrid_indices = find_nearest_grid(
    funchal_loc[0], funchal_loc[1],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)
# (4, 25)
# [inversion_height_madeira3d.lat[nearestgrid_indices[0], nearestgrid_indices[1]].values]
# [inversion_height_madeira3d.lon[nearestgrid_indices[0], nearestgrid_indices[1]].values]


#### input radiosonde data
funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')
date = pd.DatetimeIndex(np.unique(funchal_df_drvd.date))

i = 1549
altitude = funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
    'calculated_height'].values
temperature = funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
    'temperature'].values
dinv = inversion_layer(temperature, altitude, topo = 58)
teminv = temperature[np.where(altitude == dinv)]

#### input correspongding modeling data
madeira3d_sim = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd' + \
    str(date[i])[0:4] + str(date[i])[5:7] + str(date[i])[8:10] + \
    str(date[i])[11:13] + '0000.nc')
altitude_sim = (
    (madeira3d_sim.vcoord[:-1] + madeira3d_sim.vcoord[1:])/2).values
temperature_sim = madeira3d_sim.T[
    0, :, nearestgrid_indices[0], nearestgrid_indices[1]].values
dinv_sim = inversion_layer(temperature_sim[::-1], altitude_sim[::-1])
teminv_sim = temperature_sim[np.where(altitude_sim == dinv_sim)]

#### plot
# outputfile = 'figures/00_test/trial.png'
outputfile = 'figures/10_validation2obs/10_04_radiosonde/10_04.0.3 vertical sounding profile at Funchal and in model simulation 2010080612.png'
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

plt_line = ax.plot(
    temperature, altitude, '.-', color='red', lw=0.5,
    markersize=2.5, alpha=0.5)
plt_line2 = ax.plot(
    temperature_sim, altitude_sim, '.-', color='black', lw=0.5,
    markersize=2.5, alpha=0.5)
if (not np.isnan(dinv)):
    plt_scatter = ax.scatter(
        teminv, dinv, marker='s', s=12, c = 'red', zorder = 10)
    plt_scatter2 = ax.scatter(
        teminv_sim, dinv_sim, marker='s', s=12, c='black', zorder=10)
plt_text = ax.text(
    275, 4500,
    str(date[i])[0:10] + ' ' + str(date[i])[11:13] + ':00 UTC',)
ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
# ax.set_yticklabels(np.arange(0, 5.1, 0.5))
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xticks(np.arange(245, 305.1, 10))
# ax.set_xticklabels(np.arange(245, 305.1, 10, dtype='int'))
ax.set_ylim(0, 5000)
ax.set_xlim(245, 305)
ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
fig.savefig(outputfile)


'''
hour12indices = np.where(pd.DatetimeIndex(time).hour == 12)
time[hour12indices]
inversion_height_madeira3d.inversion_height[
    hour12indices[0], nearestgrid_indices[0], nearestgrid_indices[1]]

madeira3d_filelist = \
    ['/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd' +
     str(i)[0:4] + str(i)[5:7] + str(i)[8:10] + str(i)[11:13] + '0000.nc'
     for i in date]
madeira3d_filelist = np.array(sorted(glob.glob(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20*[0-9].nc')))
madeira3d_sim = xr.open_mfdataset(
    madeira3d_filelist, data_vars='minimal', coords='minimal', compat="override")

'''
# endregion
# =============================================================================


# =============================================================================
# region animate vertical profile at grid nearest to funchal and radiosonde

#### Find nearest grid
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist[0:2],
    data_vars='minimal', coords='minimal', compat='override')
funchal_loc = np.array([32.6333, -16.9000])
nearestgrid_indices = find_nearest_grid(
    funchal_loc[0], funchal_loc[1],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)

#### input radiosonde data
funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')
date = pd.DatetimeIndex(np.unique(funchal_df_drvd.date))

#### plot
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
ims = []

for i in range(len(date)):  # range(30):  # range(len(date)): #
    # i = 26
    #### input radiosounding data
    altitude = funchal_df_drvd.iloc[
        np.where(funchal_df_drvd.date == date[i])[0]][
        'calculated_height'].values
    temperature = funchal_df_drvd.iloc[
        np.where(funchal_df_drvd.date == date[i])[0]][
        'temperature'].values
    dinv = inversion_layer(temperature, altitude, topo=58)
    teminv = temperature[np.where(altitude == dinv)]
    
    #### input correspongding modeling data
    madeira3d_sim = xr.open_dataset(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd' +
        str(date[i])[0:4] + str(date[i])[5:7] + str(date[i])[8:10] +
        str(date[i])[11:13] + '0000.nc')
    altitude_sim = (
        (madeira3d_sim.vcoord[:-1] + madeira3d_sim.vcoord[1:])/2).values
    temperature_sim = madeira3d_sim.T[
        0, :, nearestgrid_indices[0], nearestgrid_indices[1]].values
    dinv_sim = inversion_layer(temperature_sim[::-1], altitude_sim[::-1])
    teminv_sim = temperature_sim[np.where(altitude_sim == dinv_sim)]
    
    #### plot them
    plt_line = ax.plot(
        temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
    plt_line2 = ax.plot(
        temperature_sim, altitude_sim, '.-',
        color='gray', lw=0.5, markersize=2.5)
    if (not np.isnan(dinv)):
        plt_scatter = ax.scatter(teminv, dinv, s=5, c='red', zorder=10)
        plt_scatter2 = ax.scatter(
            teminv_sim, dinv_sim, s=5, c='blue', zorder=10)
    plt_text = ax.text(
        265, 4500,
        '#' + str(i) + '   ' +
        str(date[i])[0:10] + ' ' + str(date[i])[11:13] + ':00 UTC',)
    
    if (not np.isnan(dinv)):
        ims.append(plt_line + plt_line2 + [plt_scatter, plt_scatter2, plt_text])
    else:
        ims.append(plt_line + plt_line2 + [plt_text])
    print(str(i) + '/' + str(len(date)))


ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xticks(np.arange(245, 305.1, 10))
ax.set_ylim(0, 5000)
ax.set_xlim(245, 305)
ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.3 vertical sounding profile at Funchal and in model simulation 2006_2015.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

# endregion
# =============================================================================


# =============================================================================
# region plot inversion height Funchal in sim_obs in 2010

#### Import grid data and find nearest grid
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist,
    data_vars='minimal', coords='minimal', compat='override')
funchal_loc = np.array([32.6333, -16.9000])
nearestgrid_indices = find_nearest_grid(
    funchal_loc[0], funchal_loc[1],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)

# store the nearest grid data
inversion_height_nearestgrid = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[
        :, nearestgrid_indices[0], nearestgrid_indices[1]
    ].values,
    index = inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)

#### import radiosonde data
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')
inversion_height_nearestgrid_subset = inversion_height_nearestgrid.loc[
    funchal_inversion_height.index].copy()

indices2010 = np.where(pd.DatetimeIndex(
    funchal_inversion_height.index).year == 2010)
month_end2010 = pd.date_range("2010-01-01", '2010-12-31', freq="M")

#### plot
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

# plt_mean1, = ax.plot(
#     funchal_inversion_height.index[indices2010],
#     funchal_inversion_height.values[indices2010],
#     '.-', markersize=2.5,
#     linewidth=0.5, color='black',)
plt_mean2, = ax.plot(
    inversion_height_nearestgrid_subset.index[indices2010],
    inversion_height_nearestgrid_subset.values[indices2010],
    '.-', markersize=2.5,
    linewidth=0.5, color='gray',)

ax.set_xticks(month_end2010)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xlabel('Year 2010', size=10)
ax.set_ylabel("Inversion height [km]", size=10)
# ax.set_ylim(0, 5000)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.4 inversion height in obs_and_sim in 2010.png',
    dpi=600)

'''
# check
inversion_height_madeira3d.inversion_height[200, 4, 25].values
inversion_height_nearestgrid.iloc[200]

(inversion_height_nearestgrid_subset.index == funchal_inversion_height.index).all()
'''
# endregion
# =============================================================================


# =============================================================================
# region climatology of inversion height in sim_obs

#### import radiosonde data
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')

#### import simulation data
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist,
    data_vars='minimal', coords='minimal', compat='override')

#### Extract data closest to Funchal
funchal_loc = np.array([32.6333, -16.9000])
nearestgrid_indices = find_nearest_grid(
    funchal_loc[0], funchal_loc[1],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)
inversion_height_nearestgrid = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[
        :, nearestgrid_indices[0], nearestgrid_indices[1]
    ].values,
    index=inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)
inversion_height_nearestgrid_subset = inversion_height_nearestgrid.loc[
    funchal_inversion_height.index].copy()

#### Extract data closest to small ellipse center O
small_e_center = [
    center_madeira[0] + 2 * radius_madeira[0] * \
    np.cos(np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + 2 * radius_madeira[0] * \
    np.sin(np.deg2rad(angle_deg_madeira))]
v = radius_madeira/0.7
nearestgrid_indices_ec = find_nearest_grid(
    small_e_center[1], small_e_center[0],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)
# (33, 31)
inversion_height_nearestgrid_ec = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[
        :, nearestgrid_indices_ec[0], nearestgrid_indices_ec[1]
    ].values,
    index=inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)
inversion_height_nearestgrid_ec_subset = inversion_height_nearestgrid_ec.loc[
    funchal_inversion_height.index].copy()

#### Extract data closest to small ellipse center O
small_e_center2 = [
    center_madeira[0] - \
    2 * radius_madeira[0] * np.cos(np.deg2rad(angle_deg_madeira)) * 0.9 - \
    radius_madeira[1] / 3 * np.sin(np.deg2rad(angle_deg_madeira)),
    center_madeira[1] - \
    2 * radius_madeira[0] * np.sin(np.deg2rad(angle_deg_madeira)) * 0.9 + \
    radius_madeira[1] / 3 * np.cos(np.deg2rad(angle_deg_madeira)),]
nearestgrid_indices_ec2 = find_nearest_grid(
    small_e_center2[1], small_e_center2[0],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)
# (36, 23)
inversion_height_nearestgrid_ec2 = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[
        :, nearestgrid_indices_ec2[0], nearestgrid_indices_ec2[1]
    ].values,
    index=inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)
inversion_height_nearestgrid_ec_subset2 = inversion_height_nearestgrid_ec2.loc[
    funchal_inversion_height.index].copy()

#### create dataframe to store the data
monthly_inversion_height = pd.DataFrame(
    data=np.zeros((len(month), 15)),
    index=month,
    columns=['mean_inv_height', 'std_inv_height',
             'mean_inv_height_sim', 'std_inv_height_sim',
             'mean_inv_height_sim_all', 'std_inv_height_sim_all',
             'mean_inv_height_ec', 'std_inv_height_ec',
             'mean_inv_height_ec_all', 'std_inv_height_ec_all',
             'mean_inv_height_ec2', 'std_inv_height_ec2',
             'mean_inv_height_ec_all2', 'std_inv_height_ec_all2',
             'rate_of_inversion_base',
             ],)

#### Extract data from small ellipse
lon = np.ma.array(inversion_height_madeira3d.lon.values)
lat = np.ma.array(inversion_height_madeira3d.lat.values)
from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1

# se1, 73
mask_se = ellipse(
    small_e_center[0], small_e_center[1],
    radius_madeira[0] / 3, radius_madeira[1] / 3,
    np.deg2rad(angle_deg_madeira), lon, lat
    )
monthly_inversion_height_se = pd.DataFrame(
    data=np.zeros((len(month), 4)),
    index=month,
    columns=['mean_inv_height_se', 'std_inv_height_se',
             'mean_inv_height_se_all', 'std_inv_height_se_all',
             ],)
# se2
mask_se2 = ellipse(
    small_e_center2[0], small_e_center2[1],
    radius_madeira[0] / 3, radius_madeira[1] / 3,
    np.deg2rad(angle_deg_madeira), lon, lat
    )
monthly_inversion_height_se2 = pd.DataFrame(
    data=np.zeros((len(month), 4)),
    index=month,
    columns=['mean_inv_height_se2', 'std_inv_height_se2',
             'mean_inv_height_se_all2', 'std_inv_height_se_all2',
             ],)


for i in range(len(month)):
    # i = 0
    monthly_index = np.where(funchal_inversion_height.index.month == i+1)
    # np.unique(funchal_inversion_height.index.month[monthly_index])
    monthly_height = funchal_inversion_height.iloc[
        monthly_index]['inversion_height']
    monthly_height_sim = inversion_height_nearestgrid_subset.iloc[
        monthly_index]['inversion_height']
    monthly_height_ec = inversion_height_nearestgrid_ec_subset.iloc[
        monthly_index]['inversion_height']
    monthly_height_ec2 = inversion_height_nearestgrid_ec_subset2.iloc[
        monthly_index]['inversion_height']
    # len(np.where((monthly_height > 58) & (monthly_height <= 3500))
    #     [0]) / len(monthly_index[0])
    # len(np.where((monthly_height == 58))[0])
    # len(np.where((monthly_height >3500))[0])
    
    monthly_index_all = np.where(
        inversion_height_nearestgrid.index.month == i+1)
    # np.unique(inversion_height_nearestgrid.index.month[monthly_index_all])
    monthly_height_all = inversion_height_nearestgrid.iloc[
        monthly_index_all]['inversion_height']
    monthly_height_ec_all = inversion_height_nearestgrid_ec.iloc[
        monthly_index_all]['inversion_height']
    monthly_height_ec_all2 = inversion_height_nearestgrid_ec2.iloc[
        monthly_index_all]['inversion_height']
    
    # Extract data within ellipse
    monthly_height_se = inversion_height_madeira3d.inversion_height.loc[
        funchal_inversion_height.index[monthly_index],
    ].values[:, mask_se].flatten()
    monthly_height_se2 = inversion_height_madeira3d.inversion_height.loc[
        funchal_inversion_height.index[monthly_index],
    ].values[:, mask_se2].flatten()
    # 19126
    # np.unique(funchal_inversion_height.index[monthly_index].month)
    # np.unique(pd.DatetimeIndex(
    #     inversion_height_madeira3d.inversion_height.loc[
    #         funchal_inversion_height.index[monthly_index],
    #         ].time.values).month)
    monthly_height_se_all = inversion_height_madeira3d.inversion_height.loc[
        inversion_height_nearestgrid.index[monthly_index_all],
    ].values[:, mask_se].flatten()
    monthly_height_se_all2 = inversion_height_madeira3d.inversion_height.loc[
        inversion_height_nearestgrid.index[monthly_index_all],
    ].values[:, mask_se2].flatten()
    # 543120
    
    # len(np.where((monthly_height_sim < 58.0) | (monthly_height_sim > 3500))[0])
    # len(np.where((monthly_height_sim >= 58) & (monthly_height_sim <= 3500))[0])
    # np.sum(np.isnan(monthly_height_sim))
    # len(monthly_height_sim)
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height'] = \
        np.mean(monthly_height[
            np.where((monthly_height != 58) & (monthly_height <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height'] = \
        np.std(monthly_height[
            np.where((monthly_height != 58) & (monthly_height <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_sim'] = \
        np.mean(monthly_height_sim[
            np.where((monthly_height_sim > 58) &
                     (monthly_height_sim <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_sim'] = \
        np.std(monthly_height_sim[
            np.where((monthly_height_sim > 58) &
                     (monthly_height_sim <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_sim_all'] = \
        np.mean(monthly_height_all[
            np.where((monthly_height_all > 58) &
                     (monthly_height_all <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_sim_all'] = \
        np.std(monthly_height_all[
            np.where((monthly_height_all > 58) &
                     (monthly_height_all <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_ec'] = \
        np.mean(monthly_height_ec[
            np.where((monthly_height_ec > 0) &
                     (monthly_height_ec <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_ec'] = \
        np.std(monthly_height_ec[
            np.where((monthly_height_ec > 0) &
                     (monthly_height_ec <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_ec_all'] = \
        np.mean(monthly_height_ec_all[
            np.where((monthly_height_ec_all > 0) &
                     (monthly_height_ec_all <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_ec_all'] = \
        np.std(monthly_height_ec_all[
            np.where((monthly_height_ec_all > 0) &
                     (monthly_height_ec_all <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_ec2'] = \
        np.mean(monthly_height_ec2[
            np.where((monthly_height_ec2 > 0) &
                     (monthly_height_ec2 <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_ec2'] = \
        np.std(monthly_height_ec2[
            np.where((monthly_height_ec2 > 0) &
                     (monthly_height_ec2 <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_ec_all2'] = \
        np.mean(monthly_height_ec_all2[
            np.where((monthly_height_ec_all2 > 0) &
                     (monthly_height_ec_all2 <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_ec_all2'] = \
        np.std(monthly_height_ec_all2[
            np.where((monthly_height_ec_all2 > 0) &
                     (monthly_height_ec_all2 <= 3500))[0]])
    
    monthly_inversion_height_se.loc[month[i], 'mean_inv_height_se'] = \
        np.mean(monthly_height_se[
            np.where((monthly_height_se > 0) &
                     (monthly_height_se <= 3500))[0]])
    monthly_inversion_height_se.loc[month[i], 'std_inv_height_se'] = \
        np.std(monthly_height_se[
            np.where((monthly_height_se > 0) &
                     (monthly_height_se <= 3500))[0]])
    
    monthly_inversion_height_se.loc[month[i], 'mean_inv_height_se_all'] = \
        np.mean(monthly_height_se_all[
            np.where((monthly_height_se_all > 0) &
                     (monthly_height_se_all <= 3500))[0]])
    monthly_inversion_height_se.loc[month[i], 'std_inv_height_se_all'] = \
        np.std(monthly_height_se_all[
            np.where((monthly_height_se_all > 0) &
                     (monthly_height_se_all <= 3500))[0]])
    
    monthly_inversion_height_se2.loc[month[i], 'mean_inv_height_se2'] = \
        np.mean(monthly_height_se2[
            np.where((monthly_height_se2 > 0) &
                     (monthly_height_se2 <= 3500))[0]])
    monthly_inversion_height_se2.loc[month[i], 'std_inv_height_se2'] = \
        np.std(monthly_height_se2[
            np.where((monthly_height_se2 > 0) &
                     (monthly_height_se2 <= 3500))[0]])
    
    monthly_inversion_height_se2.loc[month[i], 'mean_inv_height_se_all2'] = \
        np.mean(monthly_height_se_all2[
            np.where((monthly_height_se_all2 > 0) &
                     (monthly_height_se_all2 <= 3500))[0]])
    monthly_inversion_height_se2.loc[month[i], 'std_inv_height_se_all2'] = \
        np.std(monthly_height_se_all2[
            np.where((monthly_height_se_all2 > 0) &
                     (monthly_height_se_all2 <= 3500))[0]])


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 9.5]) / 2.54)

# radiosounde
plt_mean1, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height,
    '.-', markersize=2.5, linewidth=0.5, color='red',)
plt_std1, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.std_inv_height,
    '.:', markersize=2.5, linewidth=0.5, color='red',)

# grid nearest to radiosounde
plt_mean2, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height_sim,
    '.-', markersize=2.5, linewidth=0.5, color='black',)
plt_std2, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.std_inv_height_sim,
    '.:', markersize=2.5, linewidth=0.5, color='black',)
# plt_mean3, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.mean_inv_height_sim_all,
#     '.-', markersize=2.5, linewidth=0.5, color='black', alpha = 0.25)
# plt_std3, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.std_inv_height_sim_all,
#     '.:', markersize=2.5, linewidth=0.5, color='black', alpha=0.25)

# center of se1
# plt_mean4, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.mean_inv_height_ec,
#     '.-', markersize=2.5, linewidth=0.5, color='m',)
# plt_std4, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.std_inv_height_ec,
#     '.:', markersize=2.5, linewidth=0.5, color='m',)
# plt_mean5, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.mean_inv_height_ec_all,
#     '.-', markersize=2.5, linewidth=0.5, color='m', alpha=0.25)
# plt_std5, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.std_inv_height_ec_all,
#     '.:', markersize=2.5, linewidth=0.5, color='m', alpha=0.25)

# se1
plt_mean6, = ax.plot(
    monthly_inversion_height_se.index,
    monthly_inversion_height_se.mean_inv_height_se,
    '.-', markersize=2.5, linewidth=0.5, color='blue',)
plt_std6, = ax.plot(
    monthly_inversion_height_se.index,
    monthly_inversion_height_se.std_inv_height_se,
    '.:', markersize=2.5, linewidth=0.5, color='blue',)
# plt_mean7, = ax.plot(
#     monthly_inversion_height_se.index,
#     monthly_inversion_height_se.mean_inv_height_se_all,
#     '.-', markersize=2.5, linewidth=0.5, color='blue', alpha=0.25)
# plt_std7, = ax.plot(
#     monthly_inversion_height_se.index,
#     monthly_inversion_height_se.std_inv_height_se_all,
#     '.:', markersize=2.5, linewidth=0.5, color='blue', alpha=0.25)

# center of se2
# plt_mean8, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.mean_inv_height_ec2,
#     '.-', markersize=2.5, linewidth=0.5, color='m',)
# plt_std8, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.std_inv_height_ec2,
#     '.:', markersize=2.5, linewidth=0.5, color='m',)
# plt_mean9, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.mean_inv_height_ec_all2,
#     '.-', markersize=2.5, linewidth=0.5, color='m', alpha=0.25)
# plt_std9, = ax.plot(
#     monthly_inversion_height.index,
#     monthly_inversion_height.std_inv_height_ec_all2,
#     '.:', markersize=2.5, linewidth=0.5, color='m', alpha=0.25)

# se2
# plt_mean10, = ax.plot(
#     monthly_inversion_height_se2.index,
#     monthly_inversion_height_se2.mean_inv_height_se2,
#     '.-', markersize=2.5, linewidth=0.5, color='c',)
# plt_std10, = ax.plot(
#     monthly_inversion_height_se2.index,
#     monthly_inversion_height_se2.std_inv_height_se2,
#     '.:', markersize=2.5, linewidth=0.5, color='c',)
# plt_mean11, = ax.plot(
#     monthly_inversion_height_se2.index,
#     monthly_inversion_height_se2.mean_inv_height_se_all2,
#     '.-', markersize=2.5, linewidth=0.5, color='c', alpha=0.25)
# plt_std11, = ax.plot(
#     monthly_inversion_height_se2.index,
#     monthly_inversion_height_se2.std_inv_height_se_all2,
#     '.:', markersize=2.5, linewidth=0.5, color='c', alpha=0.25)

# mountain peak
plt_alt1 = plt.axhline(y=1862, color='black', linestyle='--', lw=0.5)
plt_alt2 = plt.axhline(y=1500, color='gray', linestyle='--', lw=0.5)

# legend
ax_legend = ax.legend(
    [plt_mean1, plt_mean2, plt_mean6, plt_alt1,
     plt_std1, plt_std2, plt_std6, plt_alt2],
    ['Observed mean at Funchal', 'Simulated mean near Funchal',
     'Simulated mean over ellipse $se$', 'Pico Ruivo altitude, 1862 m',
     'Observed std at Funchal', 'Simulated std near Funchal',
     'Simulated std over ellipse $se$', 'Paul da Serra altitude, 1500 m', ],
    loc = 'lower center', frameon = False, ncol = 2, fontsize = 8,
    bbox_to_anchor = (0.45, -0.45), handlelength = 1,
    columnspacing = 1)


ax.set_xticks(month)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 2.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2'))
ax.set_xlabel('Months from 2006 to 2015', size=10)
ax.set_ylabel("Monthly inversion height [km]", size=10)
ax.set_ylim(0, 2000)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.16, right=0.96, bottom=0.28, top=0.96)
# fig.savefig(
#     'figures/10_validation2obs/10_04_radiosonde/10_04.0.5 decadal monthly inversion height_obs_sim.png',
#     dpi=600)
# fig.savefig(
#     'figures/10_validation2obs/10_04_radiosonde/10_04.0.8 decadal monthly inversion height_obs_sim_all.png',
#     dpi=600)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.9 decadal monthly inversion height_obs_sim_official.png',
    dpi=600)
# fig.savefig(
#     'figures/10_validation2obs/10_04_radiosonde/10_04.0.15 decadal monthly inversion height_obs_sim_se2.png',
#     dpi=600)


'''
# topography
nc3D_Madeira_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc')
nc3D_Madeira_c.HSURF[0, nearestgrid_indices[0], nearestgrid_indices[1]].values

plt_std1 = ax.fill_between(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height - \
        monthly_inversion_height.std_inv_height,
    monthly_inversion_height.mean_inv_height + \
    monthly_inversion_height.std_inv_height,
    color='gray', alpha=0.4, edgecolor=None,)
plt_std2 = ax.fill_between(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height_sim -
    monthly_inversion_height.std_inv_height_sim,
    monthly_inversion_height.mean_inv_height_sim +
    monthly_inversion_height.std_inv_height_sim,
    color='red', alpha=0.2, edgecolor=None,)

[inversion_height_madeira3d.lat.values[
    nearestgrid_indices_e[0], nearestgrid_indices_e[1]],
 inversion_height_madeira3d.lon.values[
    nearestgrid_indices_e[0], nearestgrid_indices_e[1]],
 ]

# check
from matplotlib.patches import Ellipse
fig, ax = framework_plot1("madeira", plot_scalebar = False)
ellipse1 = Ellipse(
    center_madeira,
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse1)

ellipse_s = Ellipse(
    small_e_center,
    radius_madeira[0] * 2 / 3, radius_madeira[1] * 2 / 3,
    angle_deg_madeira, edgecolor='blue', facecolor='none', lw=0.5)
ax.add_patch(ellipse_s)
masked_se = np.ma.ones(lon.shape)
masked_se[~mask_se] = np.ma.masked
ax.pcolormesh(
    lon, lat, masked_se,
    transform=transform, cmap='viridis', zorder=-3)
ax.scatter(
    small_e_center[0], small_e_center[1], c='red', s=2.5, zorder=2)

ellipse_s2 = Ellipse(
    small_e_center2,
    radius_madeira[0] * 2 / 3, radius_madeira[1] * 2 / 3,
    angle_deg_madeira, edgecolor='c', facecolor='none', lw=0.5)
ax.add_patch(ellipse_s2)
masked_se2 = np.ma.ones(lon.shape)
masked_se2[~mask_se2] = np.ma.masked
ax.pcolormesh(
    lon, lat, masked_se2,
    transform=transform, cmap='viridis', zorder=-3)
ax.scatter(
    small_e_center2[0], small_e_center2[1], c='red', s=2.5, zorder=2)

fig.savefig('figures/00_test/trial.png')
'''
# endregion
# =============================================================================


# =============================================================================
# region calculate hourly (monthly and annual) inversion height in the nearest grid

#### input the data
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist,
    data_vars='minimal', coords='minimal', compat='override')
funchal_loc = np.array([32.6333, -16.9000])
nearestgrid_indices = find_nearest_grid(
    funchal_loc[0], funchal_loc[1],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)
inversion_height_nearestgrid = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[
        :, nearestgrid_indices[0], nearestgrid_indices[1]
    ].values,
    index=inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)

#### store hourly inversion height
hourly_inversion_height = pd.DataFrame(
    data= np.zeros((24, 26)),
    index = np.arange(0, 24, 1),
    columns=[i + j for i, j in
             zip(np.tile(['Annual'] + list(month), 2),
                 np.repeat(['_mean', '_std'], 13))])

for i in range(24):
    # i = 0
    #### Annual inversion height
    annual_index = np.where(inversion_height_nearestgrid.index.hour == i)
    # np.unique(inversion_height_nearestgrid.index.hour[annual_index])
    
    annual_height_sim = inversion_height_nearestgrid.iloc[
        annual_index]['inversion_height']
    hourly_inversion_height.loc[i, 'Annual_mean'] = np.mean(
        annual_height_sim[
            np.where((annual_height_sim > 58) &
                     (annual_height_sim <= 3500))[0]])
    hourly_inversion_height.loc[i, 'Annual_std'] = np.std(
        annual_height_sim[
            np.where((annual_height_sim > 58) &
                     (annual_height_sim <= 3500))[0]])
    
    #### Monthly inversion height
    for j in range(12):
        # j = 0
        monthly_index = np.where(
            (inversion_height_nearestgrid.index.hour == i) &
            (inversion_height_nearestgrid.index.month == j + 1))
        # np.unique(inversion_height_nearestgrid.index.hour[monthly_index])
        # np.unique(inversion_height_nearestgrid.index.month[monthly_index])
        
        monthly_height = inversion_height_nearestgrid.iloc[
            monthly_index]['inversion_height']
        # len(np.where((monthly_height < 58.0) | (monthly_height > 3500))[0])
        # len(np.where((monthly_height >= 58) & (monthly_height <= 3500))[0])
        # np.sum(np.isnan(monthly_height))
        # len(monthly_height)
        
        hourly_inversion_height.loc[i, month[j] + '_mean'] = np.mean(
            monthly_height[np.where((monthly_height > 58) &
                                    (monthly_height <= 3500))[0]])
        hourly_inversion_height.loc[i, month[j] + '_std'] = np.std(
            monthly_height[np.where((monthly_height > 58) &
                                    (monthly_height <= 3500))[0]])

hourly_inversion_height.to_pickle(
    'scratch/inversion_height/hourly_inversion_height.pkl')

'''
#### check
# import data
hourly_inversion_height = pd.read_pickle(
    'scratch/inversion_height/hourly_inversion_height.pkl')

inversion_height_madeira3d = xr.open_mfdataset(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc',
    data_vars='minimal', coords='minimal', compat='override')
inversion_height_nearestgrid = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[:, 4, 25].values,
    index=inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)

# check annual mean and std
i = 15
j = 0
annual_height = inversion_height_nearestgrid[
    inversion_height_nearestgrid.index.hour == i]
hourly_inversion_height.loc[i, 'Annual_mean']
np.mean(annual_height[(annual_height > 58) & (annual_height <=3500)])
hourly_inversion_height.loc[i, 'Annual_std']
np.std(annual_height[(annual_height > 58) & (annual_height <=3500)])

# check monthly mean and std
j = 5
monthly_height = inversion_height_nearestgrid[
    (inversion_height_nearestgrid.index.hour == i) &
    (inversion_height_nearestgrid.index.month == j + 1)]
hourly_inversion_height.loc[i, month[j] + '_mean']
np.mean(monthly_height[(monthly_height > 58) & (monthly_height <=3500)])
hourly_inversion_height.loc[i, month[j] + '_std']
np.std(monthly_height[(monthly_height > 58) & (monthly_height <=3500)])
'''
# endregion
# =============================================================================


# =============================================================================
# region calculate hourly (monthly and annual) inversion height over ellipse se

#### input the data
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist,
    data_vars='minimal', coords='minimal', compat='override')
time = pd.DatetimeIndex(inversion_height_madeira3d.time.values)

small_e_center = [
    center_madeira[0] + 2 * radius_madeira[0] *
    np.cos(np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + 2 * radius_madeira[0] *
    np.sin(np.deg2rad(angle_deg_madeira))]
lon = np.ma.array(inversion_height_madeira3d.lon.values)
lat = np.ma.array(inversion_height_madeira3d.lat.values)
from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1

mask_se = ellipse(
    small_e_center[0], small_e_center[1],
    radius_madeira[0] / 3, radius_madeira[1] / 3,
    np.deg2rad(angle_deg_madeira), lon, lat)

#### store hourly inversion height
hourly_inversion_height_se = pd.DataFrame(
    data=np.zeros((24, 26)),
    index=np.arange(0, 24, 1),
    columns=[i + j for i, j in
             zip(np.tile(['Annual'] + list(month), 2),
                 np.repeat(['_mean', '_std'], 13))])

for i in range(24):
    # i = 0
    #### Annual inversion height
    annual_index = np.where(time.hour == i)
    # np.unique(time.hour[annual_index])
    
    annual_height_sim = inversion_height_madeira3d.inversion_height.loc[
        time[annual_index],
    ].values[:, mask_se].flatten()
    # 266596
    
    hourly_inversion_height_se.loc[i, 'Annual_mean'] = np.mean(
        annual_height_sim[
            np.where((annual_height_sim > 58) &
                     (annual_height_sim <= 3500))[0]])
    hourly_inversion_height_se.loc[i, 'Annual_std'] = np.std(
        annual_height_sim[
            np.where((annual_height_sim > 58) &
                     (annual_height_sim <= 3500))[0]])
    
    #### Monthly inversion height
    for j in range(12):
        # j = 0
        monthly_index = np.where((time.hour == i) & (time.month == j + 1))
        # np.unique(time.hour[monthly_index])
        # np.unique(time.month[monthly_index])
        monthly_height_sim = inversion_height_madeira3d.inversion_height.loc[
            time[monthly_index],
        ].values[:, mask_se].flatten()
        
        # len(np.where((monthly_height_sim < 58.0) | (monthly_height_sim > 3500))[0])
        # len(np.where((monthly_height_sim >= 58) & (monthly_height_sim <= 3500))[0])
        # np.sum(np.isnan(monthly_height_sim))
        # len(monthly_height_sim)
        
        hourly_inversion_height_se.loc[i, month[j] + '_mean'] = np.mean(
            monthly_height_sim[np.where((monthly_height_sim > 58) &
                                        (monthly_height_sim <= 3500))[0]])
        hourly_inversion_height_se.loc[i, month[j] + '_std'] = np.std(
            monthly_height_sim[np.where((monthly_height_sim > 58) &
                                        (monthly_height_sim <= 3500))[0]])

hourly_inversion_height_se.to_pickle(
    'scratch/inversion_height/hourly_inversion_height_se.pkl')


'''
#### check
# import data
hourly_inversion_height_se = pd.read_pickle(
    'scratch/inversion_height/hourly_inversion_height_se.pkl')

inversion_height_madeira3d = xr.open_mfdataset(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc',
    data_vars='minimal', coords='minimal', compat='override')
time = pd.DatetimeIndex(inversion_height_madeira3d.time.values)

small_e_center = [
    center_madeira[0] + 2 * radius_madeira[0] *
    np.cos(np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + 2 * radius_madeira[0] *
    np.sin(np.deg2rad(angle_deg_madeira))]
lon = np.ma.array(inversion_height_madeira3d.lon.values)
lat = np.ma.array(inversion_height_madeira3d.lat.values)
from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1
mask_se = ellipse(
    small_e_center[0], small_e_center[1],
    radius_madeira[0] / 3, radius_madeira[1] / 3,
    np.deg2rad(angle_deg_madeira), lon, lat)

# check annual mean and std
i = 15
annual_index = np.where(time.hour == i)
annual_height_sim = inversion_height_madeira3d.inversion_height.loc[
        time[annual_index],
    ].values[:, mask_se].flatten()
hourly_inversion_height_se.loc[i, 'Annual_mean']
np.mean(
    annual_height_sim[
        np.where((annual_height_sim > 58) & (annual_height_sim <= 3500))[0]])
hourly_inversion_height_se.loc[i, 'Annual_std']
np.std(
    annual_height_sim[
        np.where((annual_height_sim > 58) & (annual_height_sim <= 3500))[0]])

# check monthly mean and std
i = 15
j = 5
monthly_index = np.where((time.hour == i) & (time.month == j + 1))
monthly_height_sim = inversion_height_madeira3d.inversion_height.loc[
            time[monthly_index],
        ].values[:, mask_se].flatten()

hourly_inversion_height_se.loc[i, month[j] + '_mean']
np.mean(
    monthly_height_sim[np.where((monthly_height_sim > 58) &
    (monthly_height_sim <= 3500))[0]])
hourly_inversion_height_se.loc[i, month[j] + '_std']
np.std(
    monthly_height_sim[np.where((monthly_height_sim > 58) &
    (monthly_height_sim <= 3500))[0]])

'''
# endregion
# =============================================================================


# =============================================================================
# region plot hourly (monthly and annually) inversion height in the nearest grid
hourly_inversion_height = pd.read_pickle(
    'scratch/inversion_height/hourly_inversion_height.pkl')
hourly_inversion_height_se = pd.read_pickle(
    'scratch/inversion_height/hourly_inversion_height_se.pkl')


#### plot annual hourly inversion height
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 9]) / 2.54)

plt_mean1, = ax.plot(
    hourly_inversion_height.index,
    hourly_inversion_height.Annual_mean,
    '.-', markersize=2.5, linewidth=0.5, color='black',)
plt_std1, = ax.plot(
    hourly_inversion_height.index,
    hourly_inversion_height.Annual_std,
    '.:', markersize=2.5, linewidth=0.5, color='black',)
plt_mean2, = ax.plot(
    hourly_inversion_height_se.index,
    hourly_inversion_height_se.Annual_mean,
    '.-', markersize=2.5, linewidth=0.5, color='blue',)
plt_std2, = ax.plot(
    hourly_inversion_height_se.index,
    hourly_inversion_height_se.Annual_std,
    '.:', markersize=2.5, linewidth=0.5, color='blue',)
# mountain peak
plt_alt1 = plt.axhline(y=1862, color='black', linestyle='--', lw=0.5)
plt_alt2 = plt.axhline(y=1500, color='gray', linestyle='--', lw=0.5)
# legend
ax_legend = ax.legend(
    [plt_mean1, plt_mean2, plt_alt1, plt_std1, plt_std2, plt_alt2],
    ['Simulated mean near Funchal', 'Simulated mean over ellipse $se$',
     'Pico Ruivo altitude, 1862 m',
     'Simulated std near Funchal', 'Simulated std over ellipse $se$',
     'Paul da Serra altitude, 1500 m', ],
    loc='lower center', frameon=False, ncol=2, fontsize=8,
    bbox_to_anchor=(0.45, -0.375), handlelength=1,
    columnspacing=1)

ax.set_xticks(np.arange(0, 24, 2))
ax.set_xticklabels(np.arange(0, 24, 2), size=8)
ax.set_yticks(np.arange(0, 2.1, 0.5) * 1000)
ax.set_yticklabels(('0', '0.5', '1', '1.5', '2'))
ax.set_xlabel('Hours (UTC) from 2006 to 2015', size=10)
ax.set_ylabel("Hourly inversion height [km]", size=10)
ax.set_xlim(0, 23)
ax.set_ylim(0, 2000)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.16, right=0.96, bottom=0.25, top=0.96)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.6 decadal hourly inversion height_sim.png',
    dpi=600)


#### plot monthly hourly inversion height
nrow = 3
ncol = 4

fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(sharex=True, sharey=True)

for j in np.arange(0, nrow):
    # j = 0
    for k in np.arange(0, ncol):
        # k = 0
        # month[j*ncol + k]
        plt_mean, = axs[j, k].plot(
            hourly_inversion_height.index,
            hourly_inversion_height.loc[:, month[j*ncol + k] + '_mean'],
            '.-', markersize=2.5,
            linewidth=0.5, color='black',)
        plt_std, = axs[j, k].plot(
            hourly_inversion_height.index,
            hourly_inversion_height.loc[:, month[j*ncol + k] + '_std'],
            '.:', markersize=2.5,
            linewidth=0.5, color='black',)
        plt_mean2, = axs[j, k].plot(
            hourly_inversion_height_se.index,
            hourly_inversion_height_se.loc[:, month[j*ncol + k] + '_mean'],
            '.-', markersize=2.5,
            linewidth=0.5, color='blue',)
        plt_std2, = axs[j, k].plot(
            hourly_inversion_height_se.index,
            hourly_inversion_height_se.loc[:, month[j*ncol + k] + '_std'],
            '.:', markersize=2.5,
            linewidth=0.5, color='blue',)
        # mountain peak
        plt_alt1 = axs[j, k].axhline(
            y=1862, color='black', linestyle='--', lw=0.5)
        plt_alt2 = axs[j, k].axhline(
            y=1500, color='gray', linestyle='--', lw=0.5)
        
        axs[j, k].set_xticks(np.arange(0, 24, 3))
        axs[j, k].set_xticklabels(np.arange(0, 24, 3))
        axs[j, k].set_yticks(np.arange(0, 1.51, 0.5) * 1000)
        axs[j, k].set_yticklabels(('0', '0.5', '1', '1.5'))
        if((j == 1) & (k == 0)):
            axs[j, k].set_ylabel("Hourly inversion height [km]")
        
        if((j == 2) & (k == 1)):
            axs[j, k].set_xlabel('Hours from 2006 to 2015')
            axs[j, k].xaxis.set_label_coords(1, -0.15)
        axs[j, k].tick_params(length=1, labelsize=10)
        axs[j, k].set_xlim(0, 23)
        axs[j, k].set_ylim(0, 2000)
        axs[j, k].grid(
            True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
        
        axs[j, k].text(
            18, 200, month[j*ncol + k], fontweight='bold', color='black')
        print(str(j*ncol + k + 1) + '/' + str(12))

ax_legend = axs[2,1].legend(
    [plt_mean, plt_std, plt_mean2, plt_std2, plt_alt1, plt_alt2],
    ['Simulated mean near Funchal', 'Simulated std near Funchal',
     'Simulated mean over ellipse $se$', 'Simulated std over ellipse $se$',
     'Pico Ruivo altitude, 1862 m', 'Paul da Serra altitude, 1500 m',],
    loc='lower center', frameon=False, ncol=3, fontsize=12,
    bbox_to_anchor=(1, -0.6), handlelength=1, columnspacing=1,)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.7 decadal monthly hourly inversion height_sim.png',
    dpi=600)

# endregion
# =============================================================================


