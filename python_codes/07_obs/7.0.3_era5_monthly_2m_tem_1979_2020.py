

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

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
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
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)


# endregion
# =============================================================================


# =============================================================================
# region monthly 2m tem from era5 from 1979 to 2020
era5_2m_tem_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_2m_tem_1979_2020.nc',
    chunks={'time': 1})

# ifinal = 48
# pre = era5_pre_1979_2020_global.tp[:ifinal, 0, :, :].values * 1000
lon = era5_2m_tem_1979_2020.longitude.values
lat = era5_2m_tem_1979_2020.latitude.values
time = pd.to_datetime(era5_2m_tem_1979_2020.time.values)

tem_release = era5_2m_tem_1979_2020.t2m[:-2, 0, :, :].values
tem_beta = era5_2m_tem_1979_2020.t2m[-2:, 1, :, :].values
tem = np.concatenate((tem_release, tem_beta))

# np.sum(np.isnan(tem_release))
# np.sum(np.isnan(tem_beta))
# np.sum(np.isnan(era5_2m_tem_1979_2020_global.t2m[-2:, 0, :, :].values))
# time[np.where(np.vstack([time.year == 2008, time.month == 1]).all(axis=0))[0][0]]

level = np.arange(287, 307.001, 0.2)
ticks = np.arange(287, 307.001, 4)

mpl.rc('font', family='Times New Roman', size=12)
nrow = 3
ncol = 4

fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            
            i_tem = np.nanmean(tem[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            
            plt_tem = axs[j, k].pcolormesh(
                lon, lat, i_tem,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('RdYlBu_r', len(level)), rasterized=True,
                transform=transform,
            )
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6, str(np.int(np.round(np.mean(i_tem), 0))),
                       size=10,color='black')
        
        axs[j, k].set_xticks(ticklabel1km_lb[0])
        axs[j, k].set_xticklabels(ticklabel1km_lb[1])
        axs[j, k].set_yticks(ticklabel1km_lb[2])
        axs[j, k].set_yticklabels(ticklabel1km_lb[3])
        axs[j, k].tick_params(length=1, labelsize=8)

        # add borders, gridlines and set extent
        axs[j, k].add_feature(coastline, lw = 0.2)
        axs[j, k].add_feature(borders, lw = 0.2)
        gl = axs[j, k].gridlines(
            crs=transform, linewidth=0.2,
            color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        axs[j, k].set_extent(extent1km_lb, crs=transform)
        
        print(str(j*ncol + k + 1) + '/' + str(12))
        print('min: ' + str(np.min(i_tem)) + ' max: ' + str(np.max(i_tem)))

cbar = fig.colorbar(
    plt_tem, ax=axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 from Jan.1979 to Oct.2020 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/07_sst/7.5.0_Monthly 2m temperature from ERA5 from 1979-01 to 2020-09.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region monthly 2m tem anomaly from era5 in 2010

era5_2m_tem_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_2m_tem_1979_2020.nc',
    chunks={'time': 1})

# ifinal = 48
# pre = era5_pre_1979_2020_global.tp[:ifinal, 0, :, :].values * 1000
lon = era5_2m_tem_1979_2020.longitude.values
lat = era5_2m_tem_1979_2020.latitude.values
time = pd.to_datetime(era5_2m_tem_1979_2020.time.values)

tem_release = era5_2m_tem_1979_2020.t2m[:-2, 0, :, :].values
tem_beta = era5_2m_tem_1979_2020.t2m[-2:, 1, :, :].values
tem = np.concatenate((tem_release, tem_beta))

# np.sum(np.isnan(tem_release))
# np.sum(np.isnan(tem_beta))
# np.sum(np.isnan(era5_2m_tem_1979_2020_global.t2m[-2:, 0, :, :].values))
# time[np.where(np.vstack([time.year == 2008, time.month == 1]).all(axis=0))[0][0]]

level = np.arange(-4, 4.001, 0.01)
ticks = np.arange(-4, 4.001, 1)

mpl.rc('font', family='Times New Roman', size=12)
nrow = 3
ncol = 4

fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            
            monthly_tem = np.nanmean(tem[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            
            monthly_tem_2010 = tem[
                (time.year == 2010) & (time.month == j*ncol + k + 1), :, :
            ].squeeze()
            
            i_tem = monthly_tem_2010 - monthly_tem
            
            plt_tem = axs[j, k].pcolormesh(
                lon, lat, i_tem,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('RdBu_r', len(level)), rasterized=True,
                transform=transform,
            )
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6, str(np.round(np.mean(i_tem), 2)),
                       size=10,color='black')
        
        axs[j, k].set_xticks(ticklabel1km_lb[0])
        axs[j, k].set_xticklabels(ticklabel1km_lb[1])
        axs[j, k].set_yticks(ticklabel1km_lb[2])
        axs[j, k].set_yticklabels(ticklabel1km_lb[3])
        axs[j, k].tick_params(length=1, labelsize=8)

        # add borders, gridlines and set extent
        axs[j, k].add_feature(coastline, lw = 0.2)
        axs[j, k].add_feature(borders, lw = 0.2)
        gl = axs[j, k].gridlines(
            crs=transform, linewidth=0.2,
            color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        axs[j, k].set_extent(extent1km_lb, crs=transform)
        
        print(str(j*ncol + k + 1) + '/' + str(12))
        print('min: ' + str(np.min(i_tem)) + ' max: ' + str(np.max(i_tem)))

cbar = fig.colorbar(
    plt_tem, ax=axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=ticks, extend='neither')
cbar.ax.set_xlabel(
    'Monthly 2m temperature anomaly [K] in ERA5 from Jan.1979 to Oct.2020 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/07_sst/7.5.1_Monthly 2m temperature anomaly from ERA5 in 2010.png', dpi=1200)


'''
'''

# endregion
# =============================================================================


# =============================================================================
# region global monthly 2m tem from era5 from 1979 to 2020

era5_2m_tem_1979_2020_global = xr.open_dataset(
    'scratch/obs/era5/monthly_2m_tem_1979_2020_global.nc',
    chunks={'time': 1})

# ifinal = 48
# pre = era5_pre_1979_2020_global.tp[:ifinal, 0, :, :].values * 1000

lon = era5_2m_tem_1979_2020_global.longitude.values
lat = era5_2m_tem_1979_2020_global.latitude.values
time = pd.to_datetime(era5_2m_tem_1979_2020_global.time.values)


tem_release = era5_2m_tem_1979_2020_global.t2m[:-2, 0, :, :].values
tem_beta = era5_2m_tem_1979_2020_global.t2m[-2:, 1, :, :].values
tem = np.concatenate((tem_release, tem_beta))

# np.sum(np.isnan(tem_release))
# np.sum(np.isnan(tem_beta))
# np.sum(np.isnan(era5_2m_tem_1979_2020_global.t2m[-2:, 0, :, :].values))
# time[np.where(np.vstack([time.year == 2008, time.month == 1]).all(axis=0))[0][0]]

level = np.arange(230, 310.001, 0.2)
ticks = np.arange(230, 310.001, 10)

mpl.rc('font', family='Times New Roman', size=12)
nrow = 6
ncol = 2


fig = plt.figure(figsize=np.array([8.8*ncol, 4.8*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.05, wspace=0.1)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            
            i_tem = np.nanmean(tem[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            
            plt_tem = axs[j, k].pcolormesh(
                lon, lat, i_tem,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('RdYlBu_r', len(level)), rasterized=True,
                transform=transform,
            )
        
        # axs[j, k].text(-14.5, 25, month[j*ncol + k],
        #                fontweight='bold', color = 'white')
        axs[j, k].set_title(month[j*ncol + k], pad=3, size=10)
        
        axs[j, k].set_xticks(ticklabel_global[0])
        axs[j, k].set_xticklabels(ticklabel_global[1])
        axs[j, k].set_yticks(ticklabel_global[2])
        axs[j, k].set_yticklabels(ticklabel_global[3])
        axs[j, k].tick_params(length=1, labelsize=8)

        # add borders, gridlines and set extent
        axs[j, k].add_feature(coastline, lw = 0.1)
        axs[j, k].add_feature(borders, lw = 0.1)
        gl = axs[j, k].gridlines(
            crs=transform, linewidth=0.1,
            color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel_global[0])
        gl.ylocator = mticker.FixedLocator(ticklabel_global[2])
        axs[j, k].set_extent(extent_global, crs=transform)
        # [0, 359.99, -90, 90]
        
        print(str(j*ncol + k + 1) + '/' + str(12))

cbar = fig.colorbar(
    plt_tem, ax=axs, orientation="horizontal",  pad=0.05,
    fraction=0.09, shrink=0.8, aspect=25, anchor=(0.5, -1.2),
    ticks=ticks, extend='both')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 from Jan.1979 to Oct.2020')

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.99)
fig.savefig(
    'figures/07_sst/7.4.0_Global monthly 2m temperature in ERA5 from 1979-01 to 2020-09.png', dpi=1200)



'''
'''
# endregion
# =============================================================================









