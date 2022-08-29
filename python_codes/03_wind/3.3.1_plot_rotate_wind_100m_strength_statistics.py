

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


######## plot

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


######## data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from metpy.calc.thermo import brunt_vaisala_frequency_squared
from metpy.cbook import get_test_data
from haversine import haversine
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve
from geopy import distance

######## add ellipse
from scipy import linalg
from sklearn import mixture
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


######## self defined

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
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
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
    transform,
    coastline,
    borders,
    center_madeira,
    angle_deg_madeira,
    radius_madeira,
)


from DEoAI_analysis.module.statistics_calculate import(
    get_statistics,
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
)


# endregion
# =============================================================================


# =============================================================================
# region plot wind statistics

wind_earth_1h_100m_strength_statistics = xr.open_dataset(
    "scratch/wind_earth/wind_earth_1h_100m_strength_statistics_2010.nc"
)

lon = wind_earth_1h_100m_strength_statistics.lon.values
lat = wind_earth_1h_100m_strength_statistics.lat.values

# set colormap level and ticks
windlevel = np.arange(4, 11.1, 0.01)
ticks = np.arange(4, 11.1, 1)


fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
    subplot_kw={'projection': transform})

plt_wind = ax.pcolormesh(
    lon, lat,
    wind_earth_1h_100m_strength_statistics.strength_quantiles[13, :, :],
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    rasterized=True)

cbar = fig.colorbar(
    plt_wind, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='both')
cbar.ax.set_xlabel("Mean wind velocity in 2010 [m/s]")

gl = ax.gridlines(crs=transform, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.add_feature(borders); ax.add_feature(coastline)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

# upstream_length = 1.4
# downstream_length = 3
# startpoint = [
#     center_madeira[1] + upstream_length * np.sin(
#     np.deg2rad(angle_deg_madeira)),
#     center_madeira[0] + upstream_length * np.cos(
#     np.deg2rad(angle_deg_madeira)),
# ]
# endpoint = [
#     center_madeira[1] - downstream_length * np.sin(
#     np.deg2rad(angle_deg_madeira)),
#     center_madeira[0] - downstream_length * np.cos(
#     np.deg2rad(angle_deg_madeira)),
# ]

# line1, = ax.plot(
#     [startpoint[1], endpoint[1]],
#     [startpoint[0], endpoint[0]],
#     lw=0.5, linestyle="-", color = 'black', zorder=2)

# ax.text(startpoint[1] + 0.1, startpoint[0], 'A1')
# ax.text(endpoint[1] + 0.1, endpoint[0], 'A2')

# startpoint = [
#     center_madeira[1] + upstream_length * np.sin(
#         np.deg2rad(77)),
#     center_madeira[0] + upstream_length * np.cos(
#         np.deg2rad(77)),
# ]
# endpoint = [
#     center_madeira[1] - downstream_length * np.sin(
#         np.deg2rad(77)),
#     center_madeira[0] - downstream_length * np.cos(
#         np.deg2rad(77)),
# ]

# line1, = ax.plot(
#     [startpoint[1], endpoint[1]],
#     [startpoint[0], endpoint[0]],
#     lw=0.5, linestyle="-", color='red', zorder=2)

ax.set_extent(extent1km_lb, crs=transform)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
# fig.savefig('figures/00_test/trial.png', dpi=300)
fig.savefig('figures/03_wind/3.0.5 Mean wind velocity in 2010.png', dpi=1200)
plt.close('all')



'''

'''
# endregion
# =============================================================================




