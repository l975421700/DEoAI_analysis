

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
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
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
# region monthly SST from satellite observations in 2010

filelist = sorted(glob.glob('scratch/obs/satellite_daily_sst/*.nc'))

satellite_daily_sst = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
)


lon = satellite_daily_sst.lon.values[3131:3375]
lat = satellite_daily_sst.lat.values[2283:2498]
time = pd.to_datetime(satellite_daily_sst.time.values)
sst = satellite_daily_sst.analysed_sst[:, 2283:2498, 3131:3375].values

level = np.arange(288, 300.01, 0.05)
ticks = np.arange(288, 300.01, 2)

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
            
            i_sst = np.mean(sst[
                time.month == j*ncol + k + 1, :, :
            ], axis=0)
            
            plt_sst = axs[j, k].pcolormesh(
                lon, lat, i_sst,
                norm=BoundaryNorm(level, ncolors=len(level), clip=False),
                cmap=cm.get_cmap('RdBu_r', len(level)), rasterized=True,
                transform=transform,
            )
        
        axs[j, k].text(-14.5, 25, month[j*ncol + k], fontweight='bold')
        
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
    plt_sst, ax=axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=ticks, extend='both')
cbar.ax.set_xlabel('Sea surface temperature from satellite observations in 2010 [K]')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
fig.savefig(
    'figures/07_sst/7.0.0_SST from satellite observations in 2010.png', dpi=1200)



'''

np.sum(np.isnan(i_sst))

# lon[3131]
# lon[3374]
# lat[2283]
# lat[2497]
# np.where(lat == 34.875)
# extent1km_lb


fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,
)

level = np.arange(290, 297.01, 0.05)
ticks = np.arange(290, 297.01, 1)

plt_dem = ax.pcolormesh(
    lon, lat,
    sst,
    norm=BoundaryNorm(level, ncolors=len(level), clip=False),
    cmap=cm.get_cmap('terrain', len(level)), rasterized=True,
    transform=transform,
)

cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.1, fraction=0.1,
    shrink=0.8, aspect=25, ticks=ticks, extend='both')
cbar.ax.set_xlabel("Sea Surface Temperature [K]")

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
plt.savefig('figures/00_test/trial.png', dpi=600)
'''
# endregion
# =============================================================================






