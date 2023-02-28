

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
# region madeira inversion layer visualization

with xr.open_dataset(
        folder_1km + '3D_Madeira/lfsd20051101000000c.nc') as madeira_3d_const:
    rlon = madeira_3d_const.rlon.values
    rlat = madeira_3d_const.rlat.values
    lon = madeira_3d_const.lon.values
    lat = madeira_3d_const.lat.values
    hhl = madeira_3d_const.HHL.squeeze().values
    hhl_level = (hhl[:-1, :, :] + hhl[1:, :, :])/2

madeira_3d_theta = xr.open_dataset(
    'scratch/3d/madeira/theta/madeira_3d_theta2010.nc')
madeira_3d_pressure = xr.open_dataset(
    'scratch/3d/madeira/pressure/madeira_3d_p2010.nc')


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)


plt_theta, = ax.plot(
    madeira_3d_theta.theta[5136 + 12, -1:38:-1, -1, -1].values,
    hhl_level[-1:38:-1, -1, -1],
    linewidth=0.25, color='red'
)

ax.set_xticks(np.arange(285, 316, 5))
ax.set_yticks(np.arange(0, 3001, 500))
ax.set_xticklabels(np.arange(285, 316, 5), size=8)
ax.set_yticklabels(np.arange(0, 3.1, 0.5), size=8)
ax.set_xlabel("Kelvin [K]", size=10)
ax.set_ylabel("Height [km]", size=10)

# output_png = 'figures/06_inversion_layer/6.0.0 inversion layer in 201008.png'
output_png = 'figures/00_test/trial.png'
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.2, top=0.99)
fig.savefig(output_png, dpi=1200)
plt.close('all')

'''
# filelist = filelist_madeira_3d = sorted(glob.glob(
#     folder_1km + '3D_Madeira/lfsd2010080[3-9]*[0-9].nc'))

# madeira_3d = xr.open_mfdataset(
#     filelist, concat_dim="time",
#     data_vars='minimal', coords='minimal', compat='override')

# madeira_3d.T; madeira_3d_theta.theta; madeira_3d_pressure.P

# plt_T, = ax.plot(
#     madeira_3d.T[0 + 12, -1:38:-1, -1, -1].values,
#     hhl_level[-1:38:-1, -1, -1],
#     linewidth=0.25, color='blue'
# )
# ax_legend = ax.legend([plt_T, plt_theta],
#                       ['Temperature', 'Potential Temperature'],
#                       loc='lower center', frameon=False, ncol=2,
#                       bbox_to_anchor=(0.5, -0.27), handlelength=1,
#                       columnspacing=1)
# for i in range(2):
#     ax_legend.get_lines()[i].set_linewidth(1)
'''
# endregion
# =============================================================================






