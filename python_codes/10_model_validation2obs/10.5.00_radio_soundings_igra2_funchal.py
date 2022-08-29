

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
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
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
# region download Funchal radio sonuding data 2006-2015

from datetime import datetime
from siphon.simplewebservice.igra2 import IGRAUpperAir

daterange = [datetime(2006, 1, 1, 0), datetime(2015, 12, 31, 23)]
station = 'POM00008522'

funchal_df, funchal_header = IGRAUpperAir.request_data(daterange, station)
funchal_df_drvd, funchal_header_drvd = IGRAUpperAir.request_data(
    daterange, station, derived=True)

funchal_df.to_pickle('scratch/radiosonde/igra2/funchal_df.pkl')
funchal_header.to_pickle('scratch/radiosonde/igra2/funchal_header.pkl')
funchal_df_drvd.to_pickle('scratch/radiosonde/igra2/funchal_df_drvd.pkl')
funchal_header_drvd.to_pickle(
    'scratch/radiosonde/igra2/funchal_header_drvd.pkl')


'''
# check

#### import newly downloaded data
funchal_df = pd.read_pickle('scratch/radiosonde/igra2/funchal_df.pkl')
funchal_header = pd.read_pickle('scratch/radiosonde/igra2/funchal_header.pkl')
funchal_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/funchal_df_drvd.pkl')
funchal_header_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_header_drvd.pkl')

#### import data using igra
import igra
data, station = igra.read.igra(
    "POM00008522", "data_source/radiosonde/POM00008522-data.txt.zip",
    return_table=True)
data_date = pd.DatetimeIndex(data.date.values)
station_date = pd.DatetimeIndex(data.date.values)
data_indices = np.where(
    (data_date.year >= 2006) & (data_date.year <= 2015))
station_indices = np.where(
    (station_date.year >= 2006) & (station_date.year <= 2015))

#### compare

pressure = data.pres[data_indices]
tem = data.temp[data_indices]
gph = data.gph[data_indices]

date2006_15 = pd.DatetimeIndex(pressure.date.values)
index_20100805 = np.where(
    (date2006_15.year == 2010) & (date2006_15.month == 8) & \
    (date2006_15.day == 5))
pressure.date[index_20100805]
pressure[index_20100805]
tem[index_20100805].values - 273.2
gph[index_20100805]


funchal_df.iloc[
    np.where(funchal_df.date == '2010-08-05T12:00:00.000000000')[0]][
        'height'].values
funchal_header.iloc[
    np.where(funchal_header.date == '2010-08-05T12:00:00.000000000')[0]][
        'number_levels'].values
funchal_df_drvd.columns
funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == '2010-08-05T12:00:00.000000000')[0]][
        'temperature'].values - 273.2
funchal_header_drvd.iloc[
    np.where(funchal_header_drvd.date == '2010-08-05T12:00:00.000000000')[0]][
        'number_levels'].values

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
# region calculate inversion height at Funchal

funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')

date = np.unique(funchal_df_drvd.date)
# pd.DatetimeIndex(date).hour.value_counts()

funchal_inversion_height = pd.DataFrame(
    data=np.repeat(0, len(date)),
    index=date,
    columns=['inversion_height'],)

for i in range(len(date)):
    # i = 0
    altitude = funchal_df_drvd.iloc[
        np.where(funchal_df_drvd.date == date[i])[0]][
            'calculated_height'].values
    temperature = funchal_df_drvd.iloc[
        np.where(funchal_df_drvd.date == date[i])[0]][
        'temperature'].values
    temperature_C = temperature - 273.2
    
    funchal_inversion_height.loc[date[i], 'inversion_height'] = \
        inversion_layer(temperature_C, altitude, topo = 58)

funchal_inversion_height.to_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')

'''
# check
funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')
date = np.unique(funchal_df_drvd.date)
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')
i = 2000
altitude = funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
        'calculated_height'].values
temperature = funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
        'temperature'].values
inversion_layer(temperature - 273.2, altitude)
funchal_inversion_height.loc[date[i], 'inversion_height']

funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]]['pressure'].values
funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
        'reported_height'].values

dinv = inversion_layer(temperature, altitude)
teminv = temperature[np.where(altitude == dinv)]
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(temperature, altitude, lw=0.5)
ax.scatter(teminv, dinv, s = 5)
fig.savefig('figures/00_test/trial.png')

'''
# endregion
# =============================================================================


# =============================================================================
# region plot inversion height at Funchal
funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')
date = np.unique(funchal_df_drvd.date)
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')

# stats.describe(
#     funchal_df_drvd.loc[funchal_df_drvd.calculated_height <=
#                         5000, 'temperature'])
# (248.3, 303.4)

# i=22
# outputfile = 'figures/00_test/trial.png'
i=25
outputfile = 'figures/00_test/trial1.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

altitude = funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
    'calculated_height'].values
temperature = funchal_df_drvd.iloc[
    np.where(funchal_df_drvd.date == date[i])[0]][
    'temperature'].values
dinv = inversion_layer(temperature, altitude)
teminv = temperature[np.where(altitude == dinv)]
plt_line = ax.plot(
    temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
if (not np.isnan(dinv)):
    plt_scatter = ax.scatter(teminv, dinv, s=5, c = 'red', zorder = 10)
plt_text = ax.text(
    265, 4500,
    '#' + str(i) + '   ' +
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
'''
# endregion
# =============================================================================


# =============================================================================
# region animate inversion height at Funchal
funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')
date = np.unique(funchal_df_drvd.date)
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')

# stats.describe(
#     funchal_df_drvd.loc[funchal_df_drvd.calculated_height <=
#                         5000, 'temperature'])
# (248.3, 303.4)

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
ims = []
for i in range(len(date)):
    # i=22
    altitude = funchal_df_drvd.iloc[
        np.where(funchal_df_drvd.date == date[i])[0]][
        'calculated_height'].values
    temperature = funchal_df_drvd.iloc[
        np.where(funchal_df_drvd.date == date[i])[0]][
        'temperature'].values
    dinv = inversion_layer(temperature, altitude)
    teminv = temperature[np.where(altitude == dinv)]
    # inversion_layer(temperature - 273.2, altitude)
    # funchal_inversion_height.loc[date[i], 'inversion_height']
    
    plt_line = ax.plot(
        temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
    if (not np.isnan(dinv)):
        plt_scatter = ax.scatter(teminv, dinv, s=5, c='red', zorder=10)
    plt_text = ax.text(
        265, 4500,
        '#' + str(i) + '   ' + \
            str(date[i])[0:10] + ' ' + str(date[i])[11:13] + ':00 UTC',)
    if (not np.isnan(dinv)):
        ims.append(plt_line + [plt_scatter, plt_text])
    else:
        ims.append(plt_line + [plt_text])
    print(str(i) + '/' + str(len(date)))

ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xticks(np.arange(245, 305.1, 10))
ax.set_xlim(245, 305)
ax.set_ylim(0, 5000)
ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.0 vertical sounding profile at Funchal 2006_2015.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


# endregion
# =============================================================================


# =============================================================================
# region plot inversion height Funchal in 2010
# funchal_header_drvd = pd.read_pickle(
#     'scratch/radiosonde/igra2/funchal_header_drvd.pkl')
# header_date = np.array(funchal_header_drvd.date)
# header_indices2010 = np.where(pd.DatetimeIndex(header_date).year == 2010)
# # header_date[header_indices2010]
# header_inv_height = funchal_header_drvd.inv_height[header_indices2010]

funchal_df_drvd = pd.read_pickle(
    'scratch/radiosonde/igra2/funchal_df_drvd.pkl')
date = np.unique(funchal_df_drvd.date)
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')

indices2010 = np.where(pd.DatetimeIndex(date).year == 2010)
date[indices2010]
month_end2010 = pd.date_range("2010-01-01", '2010-12-31', freq="M")

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_mean1, = ax.plot(
    funchal_inversion_height.index[indices2010],
    funchal_inversion_height.values[indices2010],
    '.-', markersize=2.5,
    linewidth=0.5, color='black',)
# ax.scatter(
#     funchal_inversion_height.index[indices2010],
#     funchal_inversion_height.values[indices2010],
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
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.1 inversion height in 2010.png',
    dpi=600)
# fig.savefig('figures/00_test/trial.png',)


'''
####
inversion_height = np.array(
    funchal_inversion_height.iloc[indices2010]['inversion_height'])

'''
# endregion
# =============================================================================


# =============================================================================
# region climatology of inversion height

funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')

monthly_inversion_height = pd.DataFrame(
    data=np.zeros((len(month), 2)),
    index=month,
    columns=['mean_inv_height', 'std_inv_height'],)

for i in range(len(month)):
    # i = 0
    monthly_index = np.where(funchal_inversion_height.index.month == i+1)
    # np.unique(funchal_inversion_height.index.month[monthly_index])
    
    monthly_height = funchal_inversion_height.iloc[
        monthly_index]['inversion_height']
    
    
    # len(np.where((monthly_height == 58.0) | (monthly_height > 3500))[0])
    # len(np.where((monthly_height != 58) & (monthly_height <= 3500))[0])
    # np.sum(np.isnan(monthly_height))
    # len(monthly_height)
    
    monthly_inversion_height.loc[month[i], 'mean_inv_height'] = \
        np.mean(monthly_height[
            np.where((monthly_height != 58) & (monthly_height <= 3500))[0]])
    monthly_inversion_height.loc[month[i], 'std_inv_height'] = \
        np.std(monthly_height[
            np.where((monthly_height != 58) & (monthly_height <= 3500))[0]])


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_mean1, = ax.plot(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height,
    '.-', markersize=2.5,
    linewidth=0.5, color='black',)
plt_std1 = ax.fill_between(
    monthly_inversion_height.index,
    monthly_inversion_height.mean_inv_height - \
        monthly_inversion_height.std_inv_height,
    monthly_inversion_height.mean_inv_height + \
    monthly_inversion_height.std_inv_height,
    color='gray', alpha=0.2, edgecolor=None,)
# mountain peak
plt.axhline(y=1862, color='black', linestyle='--', lw=0.5)
plt.axhline(y=1500, color='gray', linestyle='--', lw=0.5)
ax.set_xticks(month)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 3.51, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5'))
ax.set_xlabel('Months from 2006 to 2015', size=10)
ax.set_ylabel("Monthly inversion height [km]", size=10)
ax.set_ylim(0, 3500)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
fig.savefig(
    'figures/10_validation2obs/10_04_radiosonde/10_04.0.2 decadal monthly inversion height.png',
    dpi=600)

'''
plt_ds_std = ax.fill_between(
    time,
    np.mean(mean_ds_height, axis=1) - 2 * np.std(mean_ds_height, axis=1),
    np.mean(mean_ds_height, axis=1) + 2 * np.std(mean_ds_height, axis=1),
    color='gray', alpha=0.2, edgecolor=None,
    )
'''
# endregion
# =============================================================================



