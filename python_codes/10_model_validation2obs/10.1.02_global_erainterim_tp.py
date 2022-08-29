

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
# region precipitation plot over analysis region

daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1

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
annual_pre = monthly_pre.sum(axis = 0) / 10 * 1000


pre_level = np.arange(0, 600.1, 1)
pre_ticks = np.arange(0, 600.1, 100)

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
    'Annual precipitation [mm]\nERA-Interim (2006-2015)')

ax.contour(lon1, lat1, analysis_region,
           colors='red', levels=np.array([0.5]),
           linewidths=0.75, linestyles='solid'
           )

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10.1.05_Annual precipitation from ERA_Interim from 2006 to 2015.png', dpi=600)


# np.quantile(annual_pre, 0.999)
'''
# check
# ilon = 200
# ilat = 100
# itime = 30
# monmean_pre[itime, ilat, ilon]
# ei_mnth_tp.tp[itime * 8 + np.arange(0, 8), ilat, ilon].sum().values

(monmean_pre == ei_mnth_tp.tp[np.arange(0, 120) * 8 + 7].values).all()
r = ei_mnth_tp.tp.rolling(time=8).sum()

# check aggragation of precipitation
ncfile1 = xr.open_dataset(
    'data_source/era_interim_pre/ei_surface_daily_tp_200601.nc')
ncfile2 = xr.open_dataset(
    'data_source/era_interim_pre/ei_surface_synoptic_monthly_means_tp_2006.nc')
ncfile3 = xr.open_dataset(
    'data_source/era_interim_pre/ei_surface_synoptic_monthly_means_tp_2006_step12.nc')

(ncfile2.tp[3, :, :].values == ncfile3.tp[0, :, :].values).all()
(ncfile2.tp[7, :, :].values == ncfile3.tp[1, :, :].values).all()

for i in range(8):
    # i = 0
    a = ncfile1.tp[np.arange(0, 31) * 8 + i, :, :].mean(axis=0).values
    b = ncfile2.tp[i, :, :].values
    print((np.max(a), np.max(b), np.max(abs(a- b))))

'''
# endregion
# =============================================================================


# =============================================================================
# region global monthly ERA-Interim precipitation

filelist = np.array(sorted(glob.glob(
    'data_source/era_interim_pre/ei_surface_synoptic_monthly_means_tp_20??.nc'
)))
# ncfile = xr.open_dataset(
#     'data_source/era_interim_pre/ei_surface_synoptic_monthly_means_tp_2006.nc')
ei_mnth_tp = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
pre_lon = ei_mnth_tp.longitude.values
pre_lat = ei_mnth_tp.latitude.values
monmean_pre = ei_mnth_tp.tp[np.arange(0, 240) * 4 + 3].resample(
    time="1M").sum().values
monthly_pre = monmean_pre * np.tile(month_days, 10)[:, None, None] * 1000

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
            # k = 0; j = 0
            i_pre = monthly_pre[
                np.arange(0, 10) * 12 + j*ncol + k, :, :].mean(axis = 0)
            
            plt_pre = axs[j, k].pcolormesh(
                pre_lon, pre_lat, i_pre,
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
cbar.ax.set_xlabel('Monthly precipitation [mm] in ERA-Interim (2006-2015)')

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10.1.06_Global monthly precipitation from ERA_Interim from 2006 to 2015.png', dpi=600)


# endregion
# =============================================================================


