

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

# region specify common conditions
pre_level = np.arange(0, 100.001, 0.5)
pre_ticks = np.arange(0, 100.01, 20)
mpl.rc('font', family='Times New Roman', size=12)
nrow = 3
ncol = 4

######## create a mask for the area outside analysis region
# create rectangle for analysis region
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

# endregion
# =============================================================================


# =============================================================================
# region ERA5 monthly precipitation plot

######## import ERA5 pre
era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
pre_lon = era5_pre_1979_2020.longitude.values
pre_lat = era5_pre_1979_2020.latitude.values
pre_time = pd.to_datetime(era5_pre_1979_2020.time.values)
pre = np.concatenate((
    era5_pre_1979_2020.tp[:-2, 0, :, :].values,
    era5_pre_1979_2020.tp[-2:, 1, :, :].values)) * 1000

######## calculate monthly mean precipitation
mean_monthly_pre = np.zeros((12, pre.shape[1], pre.shape[2]))
for i in np.arange(0, 12):
    mean_monthly_pre[i, :, :] = month_days[i] * np.mean(
        pre[(pre_time.month == i + 1) & (pre_time.year > 2005) & (pre_time.year < 2016), :, :], axis=0)

######## create a mask to calculate domain average
pre_lon2, pre_lat2 = np.meshgrid(pre_lon, pre_lat)
pre_coors = np.hstack((pre_lon2.reshape(-1, 1), pre_lat2.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(
    pre_lon2.shape[0], pre_lon2.shape[1])

######## plot
fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            # j = 0; k = 0
            i_pre = mean_monthly_pre[j*ncol + k, :, :]
            
            plt_pre = axs[j, k].pcolormesh(
                pre_lon, pre_lat, i_pre,
                norm=BoundaryNorm(
                    pre_level, ncolors=len(pre_level), clip=False),
                cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
                transform=transform,)
            plt_mask = axs[j, k].contourf(
                mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6,
                       str(int(np.round(np.mean(i_pre[pre_mask]), 0))),
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
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in ERA5 from 2006 to 2015 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_05_monthly_pre/10_05.0.0_Monthly_pre_ERA5_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region IMERG monthly precipitation plot

######## import IMERG pre
imergm_files = np.array(sorted(glob.glob(
    'scratch/obs/gpm/imerg_monthly_pre_2006_2015/*.nc4')))
imergm_pre = xr.open_mfdataset(
    imergm_files, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
pre_lon = imergm_pre.lon.values
pre_lat = imergm_pre.lat.values
timeindex = imergm_pre.time.indexes['time'].to_datetimeindex()
daycounts = np.append(np.array((timeindex[1:] - timeindex[:-1]).days), 31)
hourcounts = daycounts * 24

######## calculate monthly mean precipitation
monthly_pre = imergm_pre.precipitation.values * hourcounts[:, None, None]
mean_monthly_pre = np.zeros(
    (12, imergm_pre.precipitation.shape[1], imergm_pre.precipitation.shape[2]))
for i in range(12):
    # i = 0
    mean_monthly_pre[i, :, :] = \
        monthly_pre[np.arange(0, 10)*12 + i, :, :].mean(0)

######## create a mask to calculate domain average
pre_lon2, pre_lat2 = np.meshgrid(pre_lon, pre_lat)
pre_coors = np.hstack((pre_lon2.reshape(-1, 1), pre_lat2.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(
    pre_lon2.shape[0], pre_lon2.shape[1])

######## plot
fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            # j = 0; k = 0
            i_pre = mean_monthly_pre[j*ncol + k, :, :]
            
            plt_pre = axs[j, k].pcolormesh(
                pre_lon, pre_lat, i_pre.T,
                norm=BoundaryNorm(
                    pre_level, ncolors=len(pre_level), clip=False),
                cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
                transform=transform,)
            plt_mask = axs[j, k].contourf(
                mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6,
                       str(int(np.round(np.mean(i_pre[pre_mask.T]), 0))),
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
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in IMERG from 2006 to 2015 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_05_monthly_pre/10_05.0.1_Monthly_pre_IMERG_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region ERA-Interim monthly precipitation plot

######## import pre
filelist = np.array(sorted(glob.glob(
    'data_source/era_interim_pre/ei_surface_synoptic_monthly_means_tp_20??.nc'
)))
ei_mnth_tp = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
pre_lon = ei_mnth_tp.longitude.values
pre_lat = ei_mnth_tp.latitude.values
monmean_pre = ei_mnth_tp.tp[np.arange(0, 240) * 4 + 3].resample(
    time="1M").sum().values
monthly_pre = monmean_pre * np.tile(month_days, 10)[:, None, None] * 1000
mean_monthly_pre = np.zeros((12, monthly_pre.shape[1], monthly_pre.shape[2]))
for i in np.arange(0, 12):
    # i=0
    mean_monthly_pre[i, :, :] = np.mean(
        monthly_pre[np.arange(0, 10) * 12 + i, :, :], axis=0)

######## create a mask to calculate domain average
pre_lon2, pre_lat2 = np.meshgrid(pre_lon, pre_lat)
pre_lon2[pre_lon2 >= 180] = pre_lon2[pre_lon2 >= 180] - 360
pre_coors = np.hstack((pre_lon2.reshape(-1, 1), pre_lat2.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(
    pre_lon2.shape[0], pre_lon2.shape[1])

######## plot
fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            # j = 0; k = 0
            i_pre = mean_monthly_pre[j*ncol + k, :, :]
            
            plt_pre = axs[j, k].pcolormesh(
                pre_lon, pre_lat, i_pre,
                norm=BoundaryNorm(
                    pre_level, ncolors=len(pre_level), clip=False),
                cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
                transform=transform,)
            plt_mask = axs[j, k].contourf(
                mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6,
                       str(int(np.round(np.mean(i_pre[pre_mask]), 0))),
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
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in ERA-Interim from 2006 to 2015 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_05_monthly_pre/10_05.0.2_Monthly_pre_ERA_Interim_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region CPS12 monthly precipitation plot

######## import CPS12 pre
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
pre = daily_pre_12km_sim.daily_precipitation.values
pre_lon = daily_pre_12km_sim.lon.values
pre_lat = daily_pre_12km_sim.lat.values
pre_time = pd.to_datetime(daily_pre_12km_sim.time.values)

mean_monthly_pre = np.zeros((12, pre.shape[1], pre.shape[2]))
for i in np.arange(0, 12):
    mean_monthly_pre[i, :, :] = month_days[i] * np.mean(
        pre[(pre_time.month == i + 1), :, :], axis=0)

######## create a mask to calculate domain average
pre_coors = np.hstack((pre_lon.reshape(-1, 1), pre_lat.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(
    pre_lon.shape[0], pre_lon.shape[1])

######## plot
fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            # j = 0; k = 0
            i_pre = mean_monthly_pre[j*ncol + k, :, :]
            
            plt_pre = axs[j, k].pcolormesh(
                pre_lon, pre_lat, i_pre,
                norm=BoundaryNorm(
                    pre_level, ncolors=len(pre_level), clip=False),
                cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
                transform=transform,)
            plt_mask = axs[j, k].contourf(
                mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6,
                       str(int(np.round(np.mean(i_pre[pre_mask]), 0))),
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
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in CPS12 from 2006 to 2015 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_05_monthly_pre/10_05.0.3_Monthly_pre_CPS12_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region CRS1 monthly precipitation plot

######## import CRS1 pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
pre = daily_pre_1km_sim.daily_precipitation.values
pre_lon = daily_pre_1km_sim.lon.values
pre_lat = daily_pre_1km_sim.lat.values
pre_time = pd.to_datetime(daily_pre_1km_sim.time.values)

mean_monthly_pre = np.zeros((12, pre.shape[1], pre.shape[2]))
for i in np.arange(0, 12):
    mean_monthly_pre[i, :, :] = month_days[i] * np.mean(
        pre[(pre_time.month == i + 1), :, :], axis=0)

######## create a mask to calculate domain average
pre_coors = np.hstack((pre_lon.reshape(-1, 1), pre_lat.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(
    pre_lon.shape[0], pre_lon.shape[1])

######## plot
fig = plt.figure(figsize=np.array([5*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform},
                  sharex = True, sharey = True)

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        if True:  # k == 0 and j == 0:  #
            # j = 0; k = 0
            i_pre = mean_monthly_pre[j*ncol + k, :, :]
            
            plt_pre = axs[j, k].pcolormesh(
                pre_lon, pre_lat, i_pre,
                norm=BoundaryNorm(
                    pre_level, ncolors=len(pre_level), clip=False),
                cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
                transform=transform,)
            plt_mask = axs[j, k].contourf(
                mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k],
                       fontweight='bold', color = 'black')
        axs[j, k].text(-23.0, 24.6,
                       str(int(np.round(np.mean(i_pre[pre_mask]), 0))),
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
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in CRS1 from 2006 to 2015 (domain average in lower left)')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_05_monthly_pre/10_05.0.4_Monthly_pre_CRS1_2006_2015.png', dpi=600)

# endregion
# =============================================================================
