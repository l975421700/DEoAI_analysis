

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
mpl.rcParams['backend'] = 'Qt4Agg'  #
# mpl.get_backend()

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

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
# region download station list

import igra
stations = igra.download.stationlist('/tmp')

fig, ax = framework_plot1("global")

# plot study region
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
ax.contour(
    lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
    linewidths=0.25, linestyles='solid')

ax.scatter(stations.lon, stations.lat, s=2,
           linewidths = 0.25, facecolors='none', edgecolors='r')
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.10 global distribution of radiosonde station from igra2.png', dpi=1200)

# POM00008522  32.6333  -16.9000   58.0    FUNCHAL                        1948 2021  29536
# SPM00060018  28.3183  -16.3822  105.0    TENERIFE-GUIMAR                2002 2021  13055

# SPM00060018
# stations.loc['SPM00060018']

# endregion
# =============================================================================


# =============================================================================
# region download Tenerif (SPM00060018) radio sonuding data 2006-2015

from datetime import datetime
from siphon.simplewebservice.igra2 import IGRAUpperAir

daterange = [datetime(2006, 1, 1, 0), datetime(2015, 12, 31, 23)]
station = 'SPM00060018'

tenerif_df, tenerif_header = IGRAUpperAir.request_data(daterange, station)
tenerif_df_drvd, tenerif_header_drvd = IGRAUpperAir.request_data(
    daterange, station, derived=True)

tenerif_df.to_pickle('scratch/radiosonde/igra2/tenerif_df.pkl')
tenerif_header.to_pickle('scratch/radiosonde/igra2/tenerif_header.pkl')
tenerif_df_drvd.to_pickle('scratch/radiosonde/igra2/tenerif_df_drvd.pkl')
tenerif_header_drvd.to_pickle(
    'scratch/radiosonde/igra2/tenerif_header_drvd.pkl')


'''
# check

#### import newly downloaded data
tenerif_df = pd.read_pickle('scratch/radiosonde/igra2/tenerif_df.pkl')
tenerif_header = pd.read_pickle('scratch/radiosonde/igra2/tenerif_header.pkl')
tenerif_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/tenerif_df_drvd.pkl')
tenerif_header_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/tenerif_header_drvd.pkl')

tenerif_df.iloc[
    np.where(tenerif_df.date == '2010-08-05T12:00:00.000000000')[0]][
        'height'].values
# 'pressure', 'temperature', 'height',
tenerif_header.iloc[
    np.where(tenerif_header.date == '2010-08-05T12:00:00.000000000')[0]][
        'number_levels'].values
tenerif_df_drvd.columns
tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == '2010-08-05T12:00:00.000000000')[0]][
        'calculated_height'].values
# 'pressure', 'reported_height', 'calculated_height', 'temperature',
tenerif_header_drvd.iloc[
    np.where(tenerif_header_drvd.date == '2010-08-05T12:00:00.000000000')[0]][
        'number_levels'].values

#### import data using igra
import igra
data, station = igra.read.igra(
    "SPM00060018", "data_source/radiosonde/SPM00060018-data.txt.zip",
    return_table=True)
data_date = pd.DatetimeIndex(data.date.values)
station_date = pd.DatetimeIndex(station.date.values)
data_indices = np.where(
    (data_date.year >= 2006) & (data_date.year <= 2015))
station_indices = np.where(
    (station_date.year >= 2006) & (station_date.year <= 2015))
# pd.DatetimeIndex(np.unique(data_date)).hour.value_counts()
# np.unique(station_date)
#### compare

pressure = data.pres[data_indices]
tem = data.temp[data_indices]
gph = data.gph[data_indices]

date2006_15 = pd.DatetimeIndex(pressure.date.values)
index_20100805 = np.where(
    (date2006_15.year == 2010) & (date2006_15.month == 8) & \
    (date2006_15.day == 5) & (date2006_15.hour == 12))
pressure.date[index_20100805]
pressure[index_20100805]
tem[index_20100805].values - 273.2
gph[index_20100805]

# data1, station1 = igra.read.igra(
#     "POM00008522", "data_source/radiosonde/POM00008522-data.txt.zip",)
# unique_date = pd.DatetimeIndex(pd.unique(date))
# indices2006_15 = np.where(
#     (unique_date.year >= 2006) & (unique_date.year <= 2015))
# unique_date2006_15 = unique_date[indices2006_15]
# unique_date2006_15.hour.value_counts()

'''
# endregion
# =============================================================================


# =============================================================================
# region calculate inversion height at Tenerif

tenerif_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/tenerif_df_drvd.pkl')

date = np.unique(tenerif_df_drvd.date)
# pd.DatetimeIndex(date).hour.value_counts()

tenerif_inversion_height = pd.DataFrame(
    data=np.repeat(0, len(date)),
    index=date,
    columns=['inversion_height'],)

for i in range(len(date)):
    # i = 0
    altitude = tenerif_df_drvd.iloc[
        np.where(tenerif_df_drvd.date == date[i])[0]][
            'calculated_height'].values
    temperature = tenerif_df_drvd.iloc[
        np.where(tenerif_df_drvd.date == date[i])[0]][
        'temperature'].values
    temperature_C = temperature - 273.2
    
    tenerif_inversion_height.loc[date[i], 'inversion_height'] = \
        inversion_layer(temperature_C, altitude, topo = 111)

tenerif_inversion_height.to_pickle(
    'scratch/inversion_height/radiosonde/tenerif_inversion_height.pkl')

'''
# check
tenerif_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/tenerif_df_drvd.pkl')
date = np.unique(tenerif_df_drvd.date)
tenerif_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/tenerif_inversion_height.pkl')
i = 0
altitude = tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == date[i])[0]][
        'calculated_height'].values
temperature = tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == date[i])[0]][
        'temperature'].values
inversion_layer(temperature - 273.2, altitude, topo = 111)
tenerif_inversion_height.loc[date[i], 'inversion_height']

tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == date[i])[0]]['pressure'].values
tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == date[i])[0]][
        'reported_height'].values

dinv = inversion_layer(temperature, altitude, topo = 111)
teminv = temperature[np.where(altitude == dinv)]
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(temperature, altitude, lw=0.5)
ax.scatter(teminv, dinv, s = 5)
fig.savefig('figures/00_test/trial.png')

'''
# endregion
# =============================================================================


# =============================================================================
# region plot inversion height Tenerif in 2010

tenerif_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/tenerif_df_drvd.pkl')
date = np.unique(tenerif_df_drvd.date)
tenerif_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/tenerif_inversion_height.pkl')

indices2010 = np.where(pd.DatetimeIndex(date).year == 2010)
# date[indices2010]
month_end2010 = pd.date_range("2010-01-01", '2010-12-31', freq="M")

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_mean1, = ax.plot(
    tenerif_inversion_height.index[indices2010],
    tenerif_inversion_height.inversion_height[indices2010[0]],
    '.-', markersize=2.5,
    linewidth=0.5, color='black',)
# ax.scatter(
#     tenerif_inversion_height.index[indices2010],
#     tenerif_inversion_height.values[indices2010],
#     s=2.5, c='black')

ax.set_xticks(month_end2010)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xlabel('Year 2010', size=10)
ax.set_ylabel("Inversion height [km]", size=10)
ax.set_ylim(0, 5000)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.13 inversion height at Tenerife in 2010.png',
    dpi=600)
# fig.savefig('figures/00_test/trial.png',)


'''
####
inversion_height = np.array(
    tenerif_inversion_height.iloc[indices2010]['inversion_height'])

'''
# endregion
# =============================================================================


# =============================================================================
# region climatology of inversion height in sim_obs at Tenerif

#### import radiosonde data
tenerif_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/tenerif_inversion_height.pkl')

#### import simulation data
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/tenerif3d/inversion_height_tenerif3d_20*.nc')))
inversion_height_tenerif3d = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')

#### Extract data closest to tenerif
tenerif_loc = np.array([28.3183, -16.3822])
nearestgrid_indices = find_nearest_grid(
    tenerif_loc[0], tenerif_loc[1],
    inversion_height_tenerif3d.lat.values,
    inversion_height_tenerif3d.lon.values)
# (inversion_height_tenerif3d.lat.values[nearestgrid_indices],
#  inversion_height_tenerif3d.lon.values[nearestgrid_indices],)
inversion_height_nearestgrid = pd.DataFrame(
    data=inversion_height_tenerif3d.inversion_height[
        :, nearestgrid_indices[0], nearestgrid_indices[1]
    ].values,
    index=inversion_height_tenerif3d.time.values,
    columns=['inversion_height'],)
inversion_height_nearestgrid_subset = inversion_height_nearestgrid.loc[
    tenerif_inversion_height.index].copy()

#### create dataframe to store the data
monthly_inversion_height = pd.DataFrame(
    data=np.zeros((len(month), 6)),
    index=month,
    columns=['mean_inv_height', 'std_inv_height',
             'mean_inv_height_sim', 'std_inv_height_sim',
             'mean_inv_height_sim_all', 'std_inv_height_sim_all',
             ],)


for i in range(len(month)):
    # i = 0
    monthly_index = np.where(tenerif_inversion_height.index.month == i+1)
    # np.unique(tenerif_inversion_height.index.month[monthly_index])
    monthly_height = tenerif_inversion_height.iloc[
        monthly_index]['inversion_height']
    monthly_height_sim = inversion_height_nearestgrid_subset.iloc[
        monthly_index]['inversion_height']
    
    monthly_index_all = np.where(
        inversion_height_nearestgrid.index.month == i+1)
    # np.unique(inversion_height_nearestgrid.index.month[monthly_index_all])
    monthly_height_all = inversion_height_nearestgrid.iloc[
        monthly_index_all]['inversion_height']
    
    # len(np.where((monthly_height_sim < 58.0) | (monthly_height_sim > 3500))[0])
    # len(np.where((monthly_height_sim >= 58) & (monthly_height_sim <= 3500))[0])
    # np.sum(np.isnan(monthly_height_sim))
    # len(monthly_height_sim)
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height'] = \
        np.mean(monthly_height[
            np.where((monthly_height > 111) &
                     (monthly_height <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height'] = \
        np.std(monthly_height[
            np.where((monthly_height > 111) &
                     (monthly_height <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_sim'] = \
        np.mean(monthly_height_sim[
            np.where((monthly_height_sim > 158) &
                     (monthly_height_sim <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_sim'] = \
        np.std(monthly_height_sim[
            np.where((monthly_height_sim > 158) &
                     (monthly_height_sim <= 3500))[0]])
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height_sim_all'] = \
        np.mean(monthly_height_all[
            np.where((monthly_height_all > 158) &
                     (monthly_height_all <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height_sim_all'] = \
        np.std(monthly_height_all[
            np.where((monthly_height_all > 158) &
                     (monthly_height_all <= 3500))[0]])

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.5]) / 2.54)

plt_mean1, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height,
    '.-', markersize=2.5, linewidth=0.5, color='red',)
plt_std1, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.std_inv_height,
    '.:', markersize=2.5, linewidth=0.5, color='red',)

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

# mountain peak
# plt_alt1 = plt.axhline(y=1862, color='black', linestyle='--', lw=0.5)
# plt_alt2 = plt.axhline(y=1500, color='gray', linestyle='--', lw=0.5)

# legend
ax_legend = ax.legend(
    [plt_mean1, plt_mean2,
     plt_std1, plt_std2],
    ['Observed mean at Tenerif', 'Simulated mean near Tenerif',
     'Observed std at Tenerif', 'Simulated std near Tenerif', ],
    loc = 'lower center', frameon = False, ncol = 2, fontsize = 8,
    bbox_to_anchor = (0.45, -0.3), handlelength = 1,
    columnspacing = 1)

ax.set_xticks(month)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 2.1, 0.5) * 1000)
ax.set_yticklabels(('0', '0.5', '1', '1.5', '2'))
ax.set_xlabel('Months from 2006 to 2015', size=10)
ax.set_ylabel("Monthly inversion height [km]", size=10)
ax.set_ylim(0, 2000)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.16, right=0.96, bottom=0.2, top=0.96)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.11 decadal monthly inversion height_obs_sim at Tenerif.png',
    dpi=600)
# fig.savefig(
#     'figures/10_validation2obs/10_04_radiosonde/10_04.0.12 decadal monthly inversion height_obs_sim_all at Tenerif.png',
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
    small_e_center[0], small_e_center[1], c='blue', s=2.5, zorder=2)
fig.savefig('figures/00_test/trial.png')
'''
# endregion
# =============================================================================


# =============================================================================
# region plot vertical profile at grid nearest to tenerif and radiosonde

#### Find nearest grid
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/tenerif3d/inversion_height_tenerif3d_20*.nc')))
inversion_height_tenerif3d = xr.open_mfdataset(
    filelist[0:2], concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
tenerif_loc = np.array([28.3183, -16.3822])
nearestgrid_indices = find_nearest_grid(
    tenerif_loc[0], tenerif_loc[1],
    inversion_height_tenerif3d.lat.values,
    inversion_height_tenerif3d.lon.values)
# (29, 55)
# inversion_height_tenerif3d.lat[nearestgrid_indices].values
# inversion_height_tenerif3d.lon[nearestgrid_indices].values


#### input radiosonde data
tenerif_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/tenerif_df_drvd.pkl')
date = pd.DatetimeIndex(np.unique(tenerif_df_drvd.date))

i = 0
outputfile = 'figures/00_test/trial.png'
altitude = tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == date[i])[0]][
    'calculated_height'].values
temperature = tenerif_df_drvd.iloc[
    np.where(tenerif_df_drvd.date == date[i])[0]][
    'temperature'].values
dinv = inversion_layer(temperature, altitude, topo = 111)
teminv = temperature[np.where(altitude == dinv)]

#### input correspongding modeling data
tenerif3d_sim = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd' + \
    str(date[i])[0:4] + str(date[i])[5:7] + str(date[i])[8:10] + \
    str(date[i])[11:13] + '0000.nc')
altitude_sim = (
    (tenerif3d_sim.vcoord[:-1] + tenerif3d_sim.vcoord[1:])/2).values
temperature_sim = tenerif3d_sim.T[
    0, :, nearestgrid_indices[0], nearestgrid_indices[1]].values

dinv_sim = inversion_layer(temperature_sim[::-1], altitude_sim[::-1],
                           topo = 158)
teminv_sim = temperature_sim[np.where(altitude_sim == dinv_sim)]

#### plot
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

plt_line = ax.plot(
    temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
plt_line2 = ax.plot(
    temperature_sim, altitude_sim, '.-', color='gray', lw=0.5, markersize=2.5)
if (not np.isnan(dinv)):
    plt_scatter = ax.scatter(teminv, dinv, s=5, c = 'red', zorder = 10)
    plt_scatter2 = ax.scatter(teminv_sim, dinv_sim, s=5, c='blue', zorder=10)
plt_text = ax.text(
    265, 4500,
    '#' + str(i) + '   ' + \
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
    madeira3d_filelist, concat_dim="time", data_vars='minimal', coords='minimal', compat="override")

'''
# endregion
# =============================================================================


# =============================================================================
# region animate vertical profile at grid nearest to funchal and radiosonde

#### Find nearest grid
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/tenerif3d/inversion_height_tenerif3d_20*.nc')))
inversion_height_tenerif3d = xr.open_mfdataset(
    filelist[0:2], concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
tenerif_loc = np.array([28.3183, -16.3822])
nearestgrid_indices = find_nearest_grid(
    tenerif_loc[0], tenerif_loc[1],
    inversion_height_tenerif3d.lat.values,
    inversion_height_tenerif3d.lon.values)
# (29, 55)
# inversion_height_tenerif3d.lat[nearestgrid_indices].values
# inversion_height_tenerif3d.lon[nearestgrid_indices].values

#### input radiosonde data
tenerif_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/tenerif_df_drvd.pkl')
date = pd.DatetimeIndex(np.unique(tenerif_df_drvd.date))

#### plot
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
ims = []

for i in range(len(date)):  # np.arange(5995, 6004, 1):  # range(30): #
    # i = 6000
    #### input radiosounding data
    altitude = tenerif_df_drvd.iloc[
        np.where(tenerif_df_drvd.date == date[i])[0]][
        'calculated_height'].values
    temperature = tenerif_df_drvd.iloc[
        np.where(tenerif_df_drvd.date == date[i])[0]][
        'temperature'].values
    dinv = inversion_layer(temperature, altitude, topo=111)
    teminv = temperature[np.where(altitude == dinv)]
    
    #### input correspongding modeling data
    tenerif3d_sim = xr.open_dataset(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd' +
        str(date[i])[0:4] + str(date[i])[5:7] + str(date[i])[8:10] +
        str(date[i])[11:13] + '0000.nc')
    altitude_sim = (
        (tenerif3d_sim.vcoord[:-1] + tenerif3d_sim.vcoord[1:])/2).values
    temperature_sim = tenerif3d_sim.T[
        0, :, nearestgrid_indices[0], nearestgrid_indices[1]].values
    dinv_sim = inversion_layer(
        temperature_sim[::-1], altitude_sim[::-1], topo=158)
    teminv_sim = temperature_sim[np.where(altitude_sim == dinv_sim)]
    
    #### plot them
    plt_line = ax.plot(
        temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
    plt_line2 = ax.plot(
        temperature_sim, altitude_sim, '.-', color='gray', lw=0.5,
        markersize=2.5)
    if (not np.isnan(dinv)):
        plt_scatter = ax.scatter(teminv, dinv, s=5, c='red', zorder=10)
        plt_scatter2 = ax.scatter(
            teminv_sim, dinv_sim, s=5, c='blue', zorder=10)
    
    plt_text = ax.text(
        265, 4500,
        '#' + str(i) + '   ' + \
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
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.14 vertical sounding profile at Tenerife and in model simulation 2006_2015.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


'''
'''
# endregion
# =============================================================================

