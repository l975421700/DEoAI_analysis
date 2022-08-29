

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
# region monthly pre from era5 from 1979 to 2020

era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
'''
# check
era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020_check.grib', engine = 'cfgrib'
)
pre = era5_pre_1979_2020.tp.values * 1000
'''

lon = era5_pre_1979_2020.longitude.values
lat = era5_pre_1979_2020.latitude.values
time = pd.to_datetime(era5_pre_1979_2020.time.values)

pre_HadISST2 = era5_pre_1979_2020.tp[
    :np.where(np.vstack([time.year == 2020, time.month == 8]).all(axis=0))[0][0], 0, :, :].values
pre_OSTIA = era5_pre_1979_2020.tp[
    np.where(np.vstack([time.year == 2020, time.month == 8]).all(axis=0))[0][0]:, 1, :, :].values
pre = np.concatenate((pre_HadISST2, pre_OSTIA)) * 1000

# np.sum(np.isnan(pre_HadISST2[-1, :, :]))
# np.sum(np.isnan(pre_OSTIA[0, :, :]))
# time[np.where(np.vstack([time.year == 2008, time.month == 1]).all(axis=0))[0][0]]

level = np.arange(0, 90.001, 0.5)
ticks = np.arange(0, 90.01, 15)

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
            # k = 0; j = 0
            i_pre = month_days[j*ncol + k] * np.nanmean(pre[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            
            plt_pre = axs[j, k].pcolormesh(
                lon, lat, i_pre,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('Blues', len(level)), rasterized=True,
                transform=transform,
            )
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6, str(np.int(np.round(np.mean(i_pre), 0))),
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

cbar = fig.colorbar(
    plt_pre, ax=axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in ERA5 from 1979 to 2020 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/07_sst/7.2.0_Monthly precipitation from ERA5 from 1979-01 to 2020-09.png', dpi=600)



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region monthly pre anomaly from era5 in 2010

era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')

lon = era5_pre_1979_2020.longitude.values
lat = era5_pre_1979_2020.latitude.values
time = pd.to_datetime(era5_pre_1979_2020.time.values)
pre_HadISST2 = era5_pre_1979_2020.tp[
    :np.where(np.vstack([time.year == 2020, time.month == 8]).all(axis=0))[0][0], 0, :, :].values
pre_OSTIA = era5_pre_1979_2020.tp[
    np.where(np.vstack([time.year == 2020, time.month == 8]).all(axis=0))[0][0]:, 1, :, :].values

pre = np.concatenate((pre_HadISST2, pre_OSTIA)) * 1000

level = np.arange(-180, 180.001, 0.5)
ticks = np.arange(-180, 180.001, 60)

mpl.rc('font', family='Times New Roman', size=12)
nrow = 3
ncol = 4

fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex=True, sharey=True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            
            monthly_pre = month_days[j*ncol + k] * np.mean(pre[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            
            monthly_pre_2010 = month_days[j*ncol + k] * pre[
                (time.year == 2010) & (time.month == j*ncol + k + 1), :, :
            ].squeeze()
            
            i_pre = monthly_pre_2010 - monthly_pre
            
            plt_pre = axs[j, k].pcolormesh(
                lon, lat, i_pre,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('RdBu', len(level)), rasterized=True,
                transform=transform,
            )
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k], fontweight='bold')
        axs[j, k].text(-23.0, 24.6, str(np.int(np.round(np.mean(i_pre), 0))),
                       size=10, color='black')
        
        axs[j, k].set_xticks(ticklabel1km_lb[0])
        axs[j, k].set_xticklabels(ticklabel1km_lb[1])
        axs[j, k].set_yticks(ticklabel1km_lb[2])
        axs[j, k].set_yticklabels(ticklabel1km_lb[3])
        axs[j, k].tick_params(length=1, labelsize=8)
        
        # add borders, gridlines and set extent
        axs[j, k].add_feature(coastline, lw=0.2)
        axs[j, k].add_feature(borders, lw=0.2)
        gl = axs[j, k].gridlines(
            crs=transform, linewidth=0.2,
            color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        axs[j, k].set_extent(extent1km_lb, crs=transform)
        
        print(str(j*ncol + k + 1) + '/' + str(12))

cbar = fig.colorbar(
    plt_pre, ax=axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=ticks, extend='both')
cbar.ax.set_xlabel(
    'Monthly precipitation anomaly [mm] in ERA5 in 2010 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/07_sst/7.2.1_Monthly precipitation anomaly from ERA5 in 2010.png', dpi=600)


# endregion
# =============================================================================


# =============================================================================
# region global monthly pre from era5 from 1979 to 2020

era5_pre_1979_2020_global = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020_global.nc',
    chunks={'time': 1})

lon = era5_pre_1979_2020_global.longitude.values
lat = era5_pre_1979_2020_global.latitude.values
# time = pd.to_datetime(era5_pre_1979_2020_global.time.values)
# pre = era5_pre_1979_2020_global.tp.values * 1000
time = pd.to_datetime(era5_pre_1979_2020_global.time[324:444].values)
pre = era5_pre_1979_2020_global.tp[324:444, :, :].values * 1000

level = np.arange(0, 500.001, 0.5)
ticks = np.arange(0, 500.01, 100)

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
            i_pre = month_days[j*ncol + k] * np.nanmean(pre[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            plt_pre = axs[j, k].pcolormesh(
                lon, lat, i_pre,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('Blues', len(level)), rasterized=True,
                transform=transform,
            )
        
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
    plt_pre, ax=axs, orientation="horizontal",  pad=0.05,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -1),
    ticks=ticks, extend='max')
# cbar.ax.set_xlabel('Monthly precipitation [mm] in ERA5 (1979-2020)')
cbar.ax.set_xlabel('Monthly precipitation [mm] in ERA5 (2006-2015)')

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.99)
# fig.savefig(
#     'figures/07_sst/7.3.0_Global monthly precipitation in ERA5 from 1979-01 to 2020-09.png', dpi=600)
fig.savefig(
    'figures/07_sst/7.3.1_Global monthly precipitation in ERA5 from 2006 to 2015.png', dpi=600)



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region annual pre and 2m tem from era5 from 1979 to 2020

# import tem
era5_2m_tem_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_2m_tem_1979_2020.nc',
    chunks={'time': 1})
tem_lon = era5_2m_tem_1979_2020.longitude.values
tem_lat = era5_2m_tem_1979_2020.latitude.values
tem_time = pd.to_datetime(era5_2m_tem_1979_2020.time.values)
tem = np.concatenate((
    era5_2m_tem_1979_2020.t2m[:-2, 0, :, :].values,
    era5_2m_tem_1979_2020.t2m[-2:, 1, :, :].values))

# import pre
era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
pre_lon = era5_pre_1979_2020.longitude.values
pre_lat = era5_pre_1979_2020.latitude.values
pre_time = pd.to_datetime(era5_pre_1979_2020.time.values)
pre = np.concatenate((
    era5_pre_1979_2020.tp[:-2, 0, :, :].values,
    era5_pre_1979_2020.tp[-2:, 1, :, :].values)) * 1000


# calculate annual precipitation
i_pre = np.zeros((12, pre.shape[1], pre.shape[2]))
i_tem = np.zeros((12, tem.shape[1], tem.shape[2]))
for i in np.arange(0, 12):
    i_pre[i, :, :] = month_days[i] * np.mean(
        pre[pre_time.month == i + 1, :, :], axis=0)
    i_tem[i, :, :] = np.mean(tem[tem_time.month == i + 1, :, :], axis=0)

annual_pre = np.sum(i_pre, axis=0)
annual_tem = np.mean(i_tem, axis=0)


# level and ticks
pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)
tem_level = np.arange(17, 25.01, 0.02)
tem_ticks = np.arange(17, 25.01, 2)

mpl.rc('font', family='Times New Roman', size=10)
# plot pre
fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,
)

plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,
)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA5 (1979-2020) \n Range: (30, 606)')

# ax.text(-14, 34, '(30, 606)',
#         # fontweight='bold'
#         )
scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/07_sst/7.2.3_Annual precipitation from ERA5 from 1979-01 to 2020-09.png', dpi=600)


# plot tem
fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,
)

plt_tem = ax.pcolormesh(
    tem_lon, tem_lat, annual_tem - 273.15,
    norm=BoundaryNorm(tem_level, ncolors=len(tem_level), clip=False),
    cmap=cm.get_cmap('Reds', len(tem_level)), rasterized=True,
    transform=transform,
)
cbar = fig.colorbar(
    plt_tem, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=tem_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual 2m temperature [°C] in ERA5 (1979-2020) \n Range: (17.3, 25.4)')

# ax.text(-14.1, 34, '(17.3, 25.4)',
#         # fontweight='bold'
#         )
scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/07_sst/7.5.2_Annual 2m temperature from ERA5 from 1979-01 to 2020-09.png', dpi=600)



'''
bottom = cm.get_cmap('Reds', len(tem_level) * 2)
newcolors = np.vstack(
    (top(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2)))),
     [1, 1, 1, 1],
     bottom(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2))))))
newcmp = ListedColormap(newcolors, name='RedsBlues_r')

'''
# endregion
# =============================================================================


# =============================================================================
# region monthly pre from era5 from 2006 to 2015

era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
'''
# check
era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020_check.grib', engine = 'cfgrib'
)
pre = era5_pre_1979_2020.tp.values * 1000
'''

lon = era5_pre_1979_2020.longitude.values
lat = era5_pre_1979_2020.latitude.values
time = pd.to_datetime(era5_pre_1979_2020.time.values)

pre_HadISST2 = era5_pre_1979_2020.tp[
    :np.where(np.vstack([time.year == 2020, time.month == 8]).all(axis=0))[0][0], 0, :, :].values
pre_OSTIA = era5_pre_1979_2020.tp[
    np.where(np.vstack([time.year == 2020, time.month == 8]).all(axis=0))[0][0]:, 1, :, :].values
pre = np.concatenate((pre_HadISST2, pre_OSTIA)) * 1000

# np.sum(np.isnan(pre_HadISST2[-1, :, :]))
# np.sum(np.isnan(pre_OSTIA[0, :, :]))
# np.sum(np.isnan(pre))
# time[np.where(np.vstack([time.year == 2008, time.month == 1]).all(axis=0))[0][0]]

level = np.arange(0, 90.001, 0.5)
ticks = np.arange(0, 90.01, 15)

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
            # k = 0; j = 0
            i_pre = month_days[j*ncol + k] * np.nanmean(pre[
                (time.month == j*ncol + k + 1) & (time.year > 2005) & (time.year < 2016), :, :
            ], axis=0)
            
            plt_pre = axs[j, k].pcolormesh(
                lon, lat, i_pre,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('Blues', len(level)), rasterized=True,
                transform=transform,
            )
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6, str(np.int(np.round(np.mean(i_pre), 0))),
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

cbar = fig.colorbar(
    plt_pre, ax=axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in ERA5 from 2006 to 2015 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/07_sst/7.2.4_Monthly precipitation from ERA5 from 2006-01 to 2015-09.png', dpi=600)



'''
'''
# endregion
# =============================================================================



