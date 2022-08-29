

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


################################ precipitation over the analysis region
# =============================================================================
# =============================================================================
# region annual pre in ERA5 from 2006-2015

simplified_vers = False

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

# extent1km_lb = [-23.401758, -11.290954, 24.182158, 34.85296]
if simplified_vers:
    mask_lon = np.arange(-23.5, -11.2, 0.02)
    mask_lat = np.arange(24.1, 34.9, 0.02)
    mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
    coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
    mask = poly_path.contains_points(coors).reshape(
        mask_lon2.shape[0], mask_lon2.shape[1])
    masked = np.ones_like(mask_lon2)
    masked[mask] = np.nan

######## import ERA5 pre
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
for i in np.arange(0, 12):
    i_pre[i, :, :] = month_days[i] * np.mean(
        pre[(pre_time.month == i + 1) & (pre_time.year > 2005) & (pre_time.year < 2016), :, :], axis=0)
annual_pre = np.sum(i_pre, axis=0)

# calculate minimum and maximum precipitation
pre_lon2, pre_lat2 = np.meshgrid(pre_lon, pre_lat)
pre_coors = np.hstack((pre_lon2.reshape(-1, 1), pre_lat2.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(43, 49)


#### plot precipitation
pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)
mpl.rc('font', family='Times New Roman', size=10)

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA5 (2006-2015)\nRange: (' + \
    str(int(np.min(annual_pre[pre_mask]))) + ', ' + \
    str(int(np.max(annual_pre[pre_mask]))) + ')')

if simplified_vers:
    ax.contourf(mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.0.0_Annual_pre_ERA5_2006_2015.png'
else:
    ax.contour(lon1, lat1, analysis_region,
               colors='red', levels=np.array([0.5]),
               linewidths=0.75, linestyles='solid')
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.1.0_Annual_pre_ERA5_2006_2015_more_info.png'

scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False,)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(outputfile, dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region annual pre in IMERG from 2006-2015

simplified_vers = False

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

# extent1km_lb = [-23.401758, -11.290954, 24.182158, 34.85296]
if simplified_vers:
    mask_lon = np.arange(-23.5, -11.2, 0.02)
    mask_lat = np.arange(24.1, 34.9, 0.02)
    mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
    coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
    mask = poly_path.contains_points(coors).reshape(
        mask_lon2.shape[0], mask_lon2.shape[1])
    masked = np.ones_like(mask_lon2)
    masked[mask] = np.nan

######## import IMERG pre
imergm_files = np.array(sorted(
    glob.glob('scratch/obs/gpm/imerg_monthly_pre_2006_2015/*.nc4')))
imergm_pre = xr.open_mfdataset(
    imergm_files, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
pre_lon = imergm_pre.lon.values
pre_lat = imergm_pre.lat.values
timeindex = imergm_pre.time.indexes['time'].to_datetimeindex()
daycounts = np.append(np.array((timeindex[1:] - timeindex[:-1]).days), 31)
hourcounts = daycounts * 24
# mean monthly precipitation plot
monthly_pre = imergm_pre.precipitation.values * hourcounts[:, None, None]
mean_monthly_pre = np.zeros(
    (12, imergm_pre.precipitation.shape[1], imergm_pre.precipitation.shape[2]))
for i in range(12):
    # i=0
    mean_monthly_pre[i, :, :] = \
        monthly_pre[np.arange(0, 10)*12 + i, :, :].mean(0)
annual_pre = np.sum(mean_monthly_pre, axis=0)

# calculate minimum and maximum precipitation
pre_lon2, pre_lat2 = np.meshgrid(pre_lon, pre_lat)
pre_coors = np.hstack((pre_lon2.reshape(-1, 1), pre_lat2.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(
    pre_lon2.shape[0], pre_lon2.shape[1])


#### plot precipitation
pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)
mpl.rc('font', family='Times New Roman', size=10)

fig, ax = framework_plot("1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre.T,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in IMERG (2006-2015)\nRange: (' +
    str(int(np.min(annual_pre[pre_mask.T]))) + ', ' + \
    str(int(np.max(annual_pre[pre_mask.T]))) + ')')

if simplified_vers:
    ax.contourf(mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.0.1_Annual_pre_IMERG_2006_2015.png'
else:
    ax.contour(lon1, lat1, analysis_region,
               colors='red', levels=np.array([0.5]),
               linewidths=0.75, linestyles='solid')
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.1.1_Annual_pre_IMERG_2006_2015_more_info.png'

scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False,)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(outputfile, dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region annual pre in ERA-Interim from 2006-2015

simplified_vers = False

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

# extent1km_lb = [-23.401758, -11.290954, 24.182158, 34.85296]
if simplified_vers:
    mask_lon = np.arange(-23.5, -11.2, 0.02)
    mask_lat = np.arange(24.1, 34.9, 0.02)
    mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
    coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
    mask = poly_path.contains_points(coors).reshape(
        mask_lon2.shape[0], mask_lon2.shape[1])
    masked = np.ones_like(mask_lon2)
    masked[mask] = np.nan

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
monthly_pre = monmean_pre * np.tile(month_days, 10)[:, None, None]
annual_pre = monthly_pre.sum(axis=0) / 10 * 1000

# calculate minimum and maximum precipitation
pre_lon2, pre_lat2 = np.meshgrid(pre_lon, pre_lat)
pre_lon2[pre_lon2 >= 180] = pre_lon2[pre_lon2 >= 180] - 360
pre_coors = np.hstack((pre_lon2.reshape(-1, 1), pre_lat2.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(241, 480)


#### plot precipitation
pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)
mpl.rc('font', family='Times New Roman', size=10)

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA-Interim\n(2006-2015) Range: (' + \
    str(int(np.min(annual_pre[pre_mask]))) + ', ' + \
    str(int(np.max(annual_pre[pre_mask]))) + ')')

if simplified_vers:
    ax.contourf(mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.0.2_Annual_pre_ERA_Interim_2006_2015.png'
else:
    ax.contour(lon1, lat1, analysis_region,
               colors='red', levels=np.array([0.5]),
               linewidths=0.75, linestyles='solid')
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.1.2_Annual_pre_ERA_Interim_2006_2015_more_info.png'

scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False,)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(outputfile, dpi=600)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region annual pre in CPS12 from 2006-2015

simplified_vers = False

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

# extent1km_lb = [-23.401758, -11.290954, 24.182158, 34.85296]
if simplified_vers:
    mask_lon = np.arange(-23.5, -11.2, 0.02)
    mask_lat = np.arange(24.1, 34.9, 0.02)
    mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
    coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
    mask = poly_path.contains_points(coors).reshape(
        mask_lon2.shape[0], mask_lon2.shape[1])
    masked = np.ones_like(mask_lon2)
    masked[mask] = np.nan

######## import CPS12 pre
nc12km_second_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h_second/lffd20000101000000c.nc')
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
annual_pre = daily_pre_12km_sim.daily_precipitation.values.sum(axis=0) / 10
pre_lon = daily_pre_12km_sim.lon.values
pre_lat = daily_pre_12km_sim.lat.values

# calculate minimum and maximum precipitation
pre_coors = np.hstack((pre_lon.reshape(-1, 1), pre_lat.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(165, 165)

#### plot precipitation
pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)
mpl.rc('font', family='Times New Roman', size=10)

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in CPS12 (2006-2015)\nRange: (' +
    str(int(np.min(annual_pre[pre_mask]))) + ', ' + \
    str(int(np.max(annual_pre[pre_mask]))) + ')')

if simplified_vers:
    ax.contourf(mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.0.3_Annual_pre_CPS12_2006_2015.png'
else:
    ax.contour(lon1, lat1, analysis_region,
               colors='red', levels=np.array([0.5]),
               linewidths=0.75, linestyles='solid')
    ax.contour(pre_lon, pre_lat, nc12km_second_c.HSURF[0].values,
               colors='gray', levels=np.array([0.5]),
               linewidths=0.75, linestyles='solid')
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.1.3_Annual_pre_CPS12_2006_2015_more_info.png'

scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False,)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(outputfile, dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region annual pre in CRS1 from 2006-2015

simplified_vers = True

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

# extent1km_lb = [-23.401758, -11.290954, 24.182158, 34.85296]
if simplified_vers:
    mask_lon = np.arange(-23.5, -11.2, 0.02)
    mask_lat = np.arange(24.1, 34.9, 0.02)
    mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
    coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
    mask = poly_path.contains_points(coors).reshape(
        mask_lon2.shape[0], mask_lon2.shape[1])
    masked = np.ones_like(mask_lon2)
    masked[mask] = np.nan

######## import CRS1 pre
annual_pre = daily_pre_1km_sim.daily_precipitation.values.sum(axis=0) / 10
pre_lon = daily_pre_1km_sim.lon.values
pre_lat = daily_pre_1km_sim.lat.values

# calculate minimum and maximum precipitation
pre_coors = np.hstack((pre_lon.reshape(-1, 1), pre_lat.reshape(-1, 1)))
pre_mask = poly_path.contains_points(pre_coors).reshape(1000, 1000)

#### plot precipitation
pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)
mpl.rc('font', family='Times New Roman', size=10)

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 9.2]) / 2.54,)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in CRS1 (2006-2015)\nRange: (' +
    str(int(np.min(annual_pre[pre_mask]))) + ', ' + \
    str(int(np.max(annual_pre[pre_mask]))) + ')')

if simplified_vers:
    ax.contourf(mask_lon2, mask_lat2, masked,
                colors='white', levels=np.array([0.5, 1.5]))
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.0.4_Annual_pre_CRS1_2006_2015.png'
else:
    ax.contour(lon1, lat1, analysis_region,
               colors='red', levels=np.array([0.5]),
               linewidths=0.75, linestyles='solid')
    outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.1.4_Annual_pre_CRS1_2006_2015_more_info.png'

scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False,)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(outputfile, dpi=600)

# endregion
# =============================================================================


################################ precipitation over Madeira
# =============================================================================
# =============================================================================
# region Madeira: annual pre in ERA5 from 2006-2015

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
for i in np.arange(0, 12):
    i_pre[i, :, :] = month_days[i] * np.mean(
        pre[(pre_time.month == i + 1) & (pre_time.year > 2005) & (pre_time.year < 2016), :, :], axis=0)

annual_pre = np.sum(i_pre, axis=0)

pre_level = np.arange(0, 1200.1, 1)
pre_ticks = np.arange(0, 1200.1, 200)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)

# plot pre
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA5 (2006-2015)')

scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.2.0_Madeira_annual_pre_ERA5_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Madeira: annual pre in IMERG from 2006-2015

# import pre
imergm_files = np.array(sorted(glob.glob(
    'scratch/obs/gpm/imerg_monthly_pre_2006_2015/*.nc4')))
imergm_pre = xr.open_mfdataset(
    imergm_files, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')

lon = imergm_pre.lon.values
lat = imergm_pre.lat.values
timeindex = imergm_pre.time.indexes['time'].to_datetimeindex()
daycounts = np.append(np.array((timeindex[1:] - timeindex[:-1]).days), 31)
hourcounts = daycounts * 24

################################ mean monthly precipitation plot
monthly_pre = imergm_pre.precipitation.values * hourcounts[:, None, None]
mean_monthly_pre = np.zeros(
    (12, imergm_pre.precipitation.shape[1], imergm_pre.precipitation.shape[2]))
for i in range(12):
    # i=0
    mean_monthly_pre[i, :, :] = \
        monthly_pre[np.arange(0, 10)*12 + i, :, :].mean(0)

annual_pre = np.sum(mean_monthly_pre, axis=0)
pre_level = np.arange(0, 1200.1, 1)
pre_ticks = np.arange(0, 1200.1, 200)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)

# plot pre
plt_pre = ax.pcolormesh(
    lon, lat, annual_pre.T,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in IMERG (2006-2015)')

scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.2.1_Madeira_annual_pre_IMERG_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Madeira: annual pre in ERA-Interim from 2006-2015

######## import ERA-interim pre
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
monthly_pre = monmean_pre * np.tile(month_days, 10)[:, None, None]
annual_pre = monthly_pre.sum(axis=0) / 10 * 1000

#### plot precipitation
pre_level = np.arange(0, 1200.1, 1)
pre_ticks = np.arange(0, 1200.1, 200)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA-Interim (2006-2015)')

scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.2.2_Madeira_annual_pre_ERA_Interim_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Madeira: annual pre in CPS12 from 2006-2015

######## import CPS12 pre
# nc12km_second_c = xr.open_dataset(
#     '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h_second/lffd20000101000000c.nc')
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
annual_pre = daily_pre_12km_sim.daily_precipitation.values.sum(axis=0) / 10
pre_lon = daily_pre_12km_sim.lon.values
pre_lat = daily_pre_12km_sim.lat.values

#### plot precipitation
pre_level = np.arange(0, 1200.1, 1)
pre_ticks = np.arange(0, 1200.1, 200)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in CPS12 (2006-2015)')

# ax.contour(pre_lon, pre_lat, nc12km_second_c.HSURF[0].values,
#            colors='gray', levels=np.array([0.5]),
#            linewidths=0.75, linestyles='solid')

scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.2.3_Madeira_annual_pre_CPS12_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Madeira: annual pre in CRS1 from 2006-2015

# import pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
annual_pre = daily_pre_1km_sim.daily_precipitation.values.sum(axis=0) / 10

pre_lon = daily_pre_1km_sim.lon.values
pre_lat = daily_pre_1km_sim.lat.values
pre_level = np.arange(0, 1200.1, 1)
pre_ticks = np.arange(0, 1200.1, 200)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)

# plot pre
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in CRS1 (2006-2015)')

scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.2.4_Madeira_annual_pre_CRS1_2006_2015.png', dpi=600)

# endregion
# =============================================================================


################################ precipitation over Canary
# =============================================================================
# =============================================================================
# region Canary: annual pre in ERA5 from 2006-2015

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
for i in np.arange(0, 12):
    i_pre[i, :, :] = month_days[i] * np.mean(
        pre[(pre_time.month == i + 1) & (pre_time.year > 2005) & (pre_time.year < 2016), :, :], axis=0)
annual_pre = np.sum(i_pre, axis=0)

pre_level = np.arange(0, 400.1, 1)
pre_ticks = np.arange(0, 400.1, 100)

fig, ax = framework_plot(
    "canary", figsize=np.array([15.6, 8]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True,)

# plot pre
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA5 (2006-2015)')

scale_bar(ax, bars=2, length=100, location=(0.7, 0.05),
          barheight=6, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.3.0_Canary_annual_pre_ERA5_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Canary: annual pre in IMERG from 2006-2015

# import pre
imergm_files = np.array(sorted(glob.glob(
    'scratch/obs/gpm/imerg_monthly_pre_2006_2015/*.nc4')))
imergm_pre = xr.open_mfdataset(
    imergm_files, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')

lon = imergm_pre.lon.values
lat = imergm_pre.lat.values
timeindex = imergm_pre.time.indexes['time'].to_datetimeindex()
daycounts = np.append(np.array((timeindex[1:] - timeindex[:-1]).days), 31)
hourcounts = daycounts * 24

monthly_pre = imergm_pre.precipitation.values * hourcounts[:, None, None]
mean_monthly_pre = np.zeros(
    (12, imergm_pre.precipitation.shape[1], imergm_pre.precipitation.shape[2]))
for i in range(12):
    # i=0
    mean_monthly_pre[i, :, :] = \
        monthly_pre[np.arange(0, 10)*12 + i, :, :].mean(0)

annual_pre = np.sum(mean_monthly_pre, axis=0)
pre_level = np.arange(0, 400.1, 1)
pre_ticks = np.arange(0, 400.1, 100)

fig, ax = framework_plot(
    "canary", figsize=np.array([15.6, 8]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
# plot pre
plt_pre = ax.pcolormesh(
    lon, lat, annual_pre.T,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in IMERG (2006-2015)')

scale_bar(ax, bars=2, length=100, location=(0.7, 0.05),
          barheight=6, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.3.1_Canary_annual_pre_IMERG_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Canary: annual pre in ERA-Interim from 2006-2015

######## import ERA-interim pre
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
monthly_pre = monmean_pre * np.tile(month_days, 10)[:, None, None]
annual_pre = monthly_pre.sum(axis=0) / 10 * 1000

#### plot precipitation
pre_level = np.arange(0, 400.1, 1)
pre_ticks = np.arange(0, 400.1, 100)

fig, ax = framework_plot(
    "canary", figsize=np.array([15.6, 8]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in ERA-Interim (2006-2015)')

scale_bar(ax, bars=2, length=100, location=(0.7, 0.05),
          barheight=6, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.3.2_Canary_annual_pre_ERA_Interim_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Canary: annual pre in CPS12 from 2006-2015

######## import CPS12 pre
# nc12km_second_c = xr.open_dataset(
#     '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h_second/lffd20000101000000c.nc')
daily_pre_12km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_12km_sim.nc")
annual_pre = daily_pre_12km_sim.daily_precipitation.values.sum(axis=0) / 10
pre_lon = daily_pre_12km_sim.lon.values
pre_lat = daily_pre_12km_sim.lat.values

#### plot precipitation
pre_level = np.arange(0, 400.1, 1)
pre_ticks = np.arange(0, 400.1, 100)

fig, ax = framework_plot(
    "canary", figsize=np.array([15.6, 8]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in CPS12 (2006-2015)')

# ax.contour(pre_lon, pre_lat, nc12km_second_c.HSURF[0].values,
#            colors='gray', levels=np.array([0.5]),
#            linewidths=0.75, linestyles='solid')

scale_bar(ax, bars=2, length=100, location=(0.7, 0.05),
          barheight=6, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.3.3_Canary_annual_pre_CPS12_2006_2015.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region Canary: annual pre in CRS1 from 2006-2015

# import pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
annual_pre = daily_pre_1km_sim.daily_precipitation.values.sum(axis=0) / 10

pre_lon = daily_pre_1km_sim.lon.values
pre_lat = daily_pre_1km_sim.lat.values
pre_level = np.arange(0, 400.1, 1)
pre_ticks = np.arange(0, 400.1, 100)

fig, ax = framework_plot(
    "canary", figsize=np.array([15.6, 8]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)

# plot pre
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in CRS1 (2006-2015)')

scale_bar(ax, bars=2, length=100, location=(0.7, 0.05),
          barheight=6, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_01_annual_pre/10_01.3.4_Canary_annual_pre_CRS1_2006_2015.png', dpi=600)

# endregion
# =============================================================================


################################ pre over Madeira in CRS1 for each year
# =============================================================================
# region mean values plot

# import pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")

time = pd.DatetimeIndex(daily_pre_1km_sim.time.values)
pre_lon = daily_pre_1km_sim.lon.values
pre_lat = daily_pre_1km_sim.lat.values
pre_level = np.arange(0, 2000.1, 1)
pre_ticks = np.arange(0, 2000.1, 400)

i = 4
outputfile = 'figures/10_validation2obs/10_01_annual_pre/10_01.4.0_Madeira_annual_pre_CRS1_20' + years[i] + '.png'
i_annual_pre = daily_pre_1km_sim.daily_precipitation[
    np.where(time.year == 2006 + i)[0], :, :].values.sum(axis=0)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
plt_pre = ax.pcolormesh(
    pre_lon, pre_lat, i_annual_pre,
    norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
    cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
    transform=transform, animated=True)
plt_text = ax.text(
    -16.8, 32.07,
    'Annual precipitation [mm] in CRS1 in 20' + years[i],
    horizontalalignment='center', )
cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')
scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(outputfile, dpi=600)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region mean values animation

# import pre
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")

time = pd.DatetimeIndex(daily_pre_1km_sim.time.values)
pre_lon = daily_pre_1km_sim.lon.values
pre_lat = daily_pre_1km_sim.lat.values
pre_level = np.arange(0, 2000.1, 1)
pre_ticks = np.arange(0, 2000.1, 400)

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=True)
ims =[]

for i in range(10):  # range(2):  #
    # i = 4
    i_annual_pre = daily_pre_1km_sim.daily_precipitation[
        np.where(time.year == 2006 + i)[0], :, :].values.sum(axis=0)
    plt_pre = ax.pcolormesh(
        pre_lon, pre_lat, i_annual_pre,
        norm=BoundaryNorm(pre_level, ncolors=len(pre_level), clip=False),
        cmap=cm.get_cmap('Blues', len(pre_level)), rasterized=True,
        transform=transform, animated=True)
    plt_text = ax.text(
        -16.8, 32.07,
        'Annual precipitation [mm] in CRS1 in 20' + years[i],
        horizontalalignment='center', )
    ims.append([plt_pre, plt_text])
    print(str(i) + '/9')

cbar = fig.colorbar(
    plt_pre, ax=ax, orientation="horizontal",  pad=0.1,
    fraction=0.12, shrink=0.8, aspect=25, anchor=(0.5, -0.6),
    ticks=pre_ticks, extend='max')

scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/10_validation2obs/10_01_annual_pre/10_01.4.1_Madeira_annual_pre_CRS1_2006_2015.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'), dpi=600)

'''
'''
# endregion
# =============================================================================

