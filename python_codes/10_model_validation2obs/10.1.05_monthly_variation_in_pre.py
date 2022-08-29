

# =============================================================================
# region import packages


# basic library
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc

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
from matplotlib.patches import Ellipse

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

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
    month_days,
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
    center_madeira,
    angle_deg_madeira,
    radius_madeira,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)

# endregion

# region create ellipse mask

from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1

# endregion
# =============================================================================


# =============================================================================
# region plot the grids used for extraction of precipitation in CRS1

# CRS1 monthly precipitation selected grids
######## import CRS1 pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
# pre = daily_pre_1km_sim.daily_precipitation.values
pre_lon_1km = daily_pre_1km_sim.lon.values
pre_lat_1km = daily_pre_1km_sim.lat.values
pre_1km_time = pd.to_datetime(daily_pre_1km_sim.time.values)

mask_e2_1km = ellipse(
    center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_1km, pre_lat_1km,
)
masked_e2_1km = np.ma.ones(pre_lon_1km.shape)
masked_e2_1km[~mask_e2_1km] = np.ma.masked
# 662 grids: 801.02 km^2

fig, ax = framework_plot1("1km_lb")

ellipse2 = Ellipse(
    [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
        center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse2)

model_lb2 = ax.contourf(
    pre_lon_1km, pre_lat_1km, masked_e2_1km,
    transform=transform, colors='gainsboro', alpha=1, zorder=-3)

fig.savefig('figures/10_validation2obs/10_07_monthly_pre_variation/10_07.0.1 selected grids in 1km simulation.png', dpi=600)



# endregion
# =============================================================================


# =============================================================================
# region plot the grids used for extraction of precipitation in CPS12

# CPS12 monthly precipitation selected grids
######## import CPS12 pre
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
pre_lon_12km = daily_pre_12km_sim.lon.values
pre_lat_12km = daily_pre_12km_sim.lat.values
pre_time = pd.to_datetime(daily_pre_12km_sim.time.values)

mask_e2_12km = ellipse(
    center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_12km, pre_lat_12km,
)
masked_e2_12km = np.ma.ones(pre_lon_12km.shape)
masked_e2_12km[~mask_e2_12km] = np.ma.masked
# 4 grids: 576 km^2

fig, ax = framework_plot1("1km_lb")

ellipse2 = Ellipse(
    [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
        center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse2)

ax.pcolormesh(pre_lon_12km, pre_lat_12km, masked_e2_12km,
              transform=transform, color='gainsboro', alpha=1, zorder=-3)

fig.savefig(
    'figures/10_validation2obs/10_07_monthly_pre_variation/10_07.0.1 selected grids in 12km simulation.png', dpi=600)


# endregion
# =============================================================================


# =============================================================================
# region plot the grids used for extraction of precipitation in ERA5

# ERA5 monthly precipitation selected grids
######## import ERA5 pre

era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
pre_lon_era5 = era5_pre_1979_2020.longitude.values
pre_lat_era5 = era5_pre_1979_2020.latitude.values

pre_lon_era5_2, pre_lat_era5_2 = np.meshgrid(pre_lon_era5, pre_lat_era5)

mask_e2_era5 = ellipse(
    center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_era5_2, pre_lat_era5_2,
)
masked_e2_era5 = np.ma.ones(pre_lon_era5_2.shape)
masked_e2_era5[~mask_e2_era5] = np.ma.masked
# 1 grid: 625 km**2

fig, ax = framework_plot1("1km_lb")

ellipse2 = Ellipse(
    [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
        center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse2)

ax.pcolormesh(pre_lon_era5_2, pre_lat_era5_2, masked_e2_era5,
              transform=transform, color='gainsboro', alpha=1, zorder=-3)

fig.savefig(
    'figures/10_validation2obs/10_07_monthly_pre_variation/10_07.0.2 selected grids in era5.png',
    dpi=600)


# endregion
# =============================================================================


# =============================================================================
# region plot the grids used for extraction of pre over Madeira in CRS1

# CRS1 monthly precipitation selected grids
######## import CRS1 pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
# pre = daily_pre_1km_sim.daily_precipitation.values
pre_lon_1km = daily_pre_1km_sim.lon.values
pre_lat_1km = daily_pre_1km_sim.lat.values
pre_1km_time = pd.to_datetime(daily_pre_1km_sim.time.values)

mask_e1_1km = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_1km, pre_lat_1km,
)
masked_e1_1km = np.ma.ones(pre_lon_1km.shape)
masked_e1_1km[~mask_e1_1km] = np.ma.masked
# 666 grids: 805.86 km^2

fig, ax = framework_plot1("1km_lb")

ellipse1 = Ellipse(
    center_madeira,
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse1)

ax.contourf(
    pre_lon_1km, pre_lat_1km, masked_e1_1km,
    transform=transform, colors='gainsboro', alpha=1, zorder=-3)

fig.savefig('figures/10_validation2obs/10_07_monthly_pre_variation/10_07.1.0 selected grids over Madeira in 1km simulation.png', dpi=600)



# endregion
# =============================================================================


# =============================================================================
# region plot the grids used for extraction of pre over Madeira in CPS12

# CPS12 monthly precipitation selected grids
######## import CPS12 pre
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
pre_lon_12km = daily_pre_12km_sim.lon.values
pre_lat_12km = daily_pre_12km_sim.lat.values

mask_e1_12km = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_12km, pre_lat_12km,
)
masked_e1_12km = np.ma.ones(pre_lon_12km.shape)
masked_e1_12km[~mask_e1_12km] = np.ma.masked
# 6 grids: 864 km^2

fig, ax = framework_plot1("1km_lb")

ellipse1 = Ellipse(
    center_madeira,
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse1)

ax.pcolormesh(pre_lon_12km, pre_lat_12km, masked_e1_12km,
              transform=transform, color='gainsboro', alpha=1, zorder=-3)

fig.savefig(
    'figures/10_validation2obs/10_07_monthly_pre_variation/10_07.1.1 selected grids over Madeira in 12km simulation.png', dpi=600)


# endregion
# =============================================================================


# =============================================================================
# region plot the grids used for extraction of pre over Madeira in ERA5

# ERA5 monthly precipitation selected grids
######## import ERA5 pre

era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
pre_lon_era5 = era5_pre_1979_2020.longitude.values
pre_lat_era5 = era5_pre_1979_2020.latitude.values

pre_lon_era5_2, pre_lat_era5_2 = np.meshgrid(pre_lon_era5, pre_lat_era5)

mask_e1_era5 = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_era5_2, pre_lat_era5_2,
)
masked_e1_era5 = np.ma.ones(pre_lon_era5_2.shape)
masked_e1_era5[~mask_e1_era5] = np.ma.masked
# 1 grid: 625 km**2

fig, ax = framework_plot1("1km_lb")

ellipse1 = Ellipse(
    center_madeira,
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse1)

ax.pcolormesh(pre_lon_era5_2, pre_lat_era5_2, masked_e1_era5,
              transform=transform, color='gainsboro', alpha=1, zorder=-3)

fig.savefig(
    'figures/10_validation2obs/10_07_monthly_pre_variation/10_07.1.2 selected grids over Madeira in era5.png',
    dpi=600)


# endregion
# =============================================================================


# =============================================================================
# region calculate the monthly precipitation

# create an array to store the results
monthly_pre = pd.DataFrame(
    data=np.zeros((120, 3)),
    index=pd.date_range(start='2006-01-01', end='2016-01-01', freq='M'),
    columns=['era5', 'CPS12', 'CRS1'])

monthly_variation = pd.DataFrame(
    data=np.zeros((12, 3)),
    index=month,
    columns=['era5', 'CPS12', 'CRS1'])

# extract precipitation in ERA5
era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
pre_lon_era5 = era5_pre_1979_2020.longitude.values
pre_lat_era5 = era5_pre_1979_2020.latitude.values
pre_lon_era5_2, pre_lat_era5_2 = np.meshgrid(pre_lon_era5, pre_lat_era5)
pre_time_era5 = pd.to_datetime(era5_pre_1979_2020.time.values)
pre_era5 = np.concatenate((
    era5_pre_1979_2020.tp[:-2, 0, :, :].values,
    era5_pre_1979_2020.tp[-2:, 1, :, :].values)) * 1000

mask_e1_era5 = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_era5_2, pre_lat_era5_2,)

# extract precipitation in CPS12
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
pre_lon_12km = daily_pre_12km_sim.lon.values
pre_lat_12km = daily_pre_12km_sim.lat.values
pre_time_12km = pd.to_datetime(daily_pre_12km_sim.time.values)
pre_12km = daily_pre_12km_sim.daily_precipitation
mask_e1_12km = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_12km, pre_lat_12km,
)

# extract precipitation in CRS1
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
pre_lon_1km = daily_pre_1km_sim.lon.values
pre_lat_1km = daily_pre_1km_sim.lat.values
pre_time_1km = pd.to_datetime(daily_pre_1km_sim.time.values)
pre_1km = daily_pre_1km_sim.daily_precipitation

mask_e1_1km = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), pre_lon_1km, pre_lat_1km,
)


for i in range(len(monthly_pre.index)):
    # i = 100
    # monthly_pre.index[i].year; monthly_pre.index[i].month
    index_era5 = np.where((pre_time_era5.year == monthly_pre.index[i].year) &
                          (pre_time_era5.month == monthly_pre.index[i].month))
    # monthly_pre.index[i]; pre_time_era5[index_era5]
    # pre_era5[index_era5, mask_e1_era5], pre_era5[325, 8, 26]
    monthly_pre.era5.iloc[i] = pre_era5[index_era5, mask_e1_era5][0][0]
    
    index_cps12 = np.where((pre_time_12km.year == monthly_pre.index[i].year) &
                           (pre_time_12km.month == monthly_pre.index[i].month))
    # pre_time_12km[index_cps12]
    monthly_pre.CPS12.iloc[i] = pre_12km[index_cps12[0],
                                         ].values[:, mask_e1_12km].mean()
    
    index_crs1 = np.where((pre_time_1km.year == monthly_pre.index[i].year) &
                          (pre_time_1km.month == monthly_pre.index[i].month))
    monthly_pre.CRS1.iloc[i] = pre_1km[index_crs1[0],
                                       ].values[:, mask_e1_1km].mean()
    
    # monthly_pre.iloc[i]
    print(str(i) + '/' + str(len(monthly_pre.index)))

for i in range(len(month)):
    # i = 5
    monthly_variation.era5[i] = monthly_pre.era5.iloc[
        np.where(monthly_pre.index.month == i + 1)
    ].mean()
    
    monthly_variation.CPS12[i] = monthly_pre.CPS12.iloc[
        np.where(monthly_pre.index.month == i + 1)
    ].mean()
    
    monthly_variation.CRS1[i] = monthly_pre.CRS1.iloc[
        np.where(monthly_pre.index.month == i + 1)
    ].mean()


# (monthly_variation * month_days[:, None]).to_pickle('data4figure4.pkl')
# monthly_variation1 = pd.read_pickle('data4figure4.pkl')

# plot
# month_days
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_era5, = ax.plot(
    monthly_variation.index, monthly_variation.era5 * month_days,
    '.-', markersize=2.5, linewidth=0.5, color='black',)
plt_cps12, = ax.plot(
    monthly_variation.index, monthly_variation.CPS12 * month_days,
    '.--', markersize=2.5, linewidth=0.5, color='black',)
plt_crs1, = ax.plot(
    monthly_variation.index, monthly_variation.CRS1 * month_days,
    '.:', markersize=2.5, linewidth=0.5, color='black',)

ax_legend = ax.legend(
    [plt_era5, plt_cps12, plt_crs1],
    ['ERA5', 'CPS12', 'CRS1', ],
    loc='lower center', frameon=False, ncol=3, fontsize=8,
    bbox_to_anchor=(0.5, -0.28), handlelength=2,
    columnspacing=1)

ax.set_xticks(month)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_yticklabels(np.arange(0, 101, 20))
ax.set_xlabel('Months from 2006 to 2015', size=10)
ax.set_ylabel("Monthly precipitation [mm]", size=10)
ax.set_ylim(0, 110)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.2, top=0.96)

fig.savefig(
    'figures/10_validation2obs/10_07_monthly_pre_variation/10_07.2.0 decadal monthly precipitation_obs_sim_official.png',
    dpi=600)

# endregion
# =============================================================================

