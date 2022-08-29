


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
# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

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
# region calculation

wind_earth_1h_100m_strength_direction = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

time = wind_earth_1h_100m_strength_direction.time


from eofs.standard import Eof


ddd = wind_earth_1h_100m_strength_direction.strength[0:744, :, :].values
weights_array = np.cos(np.deg2rad(
    wind_earth_1h_100m_strength_direction.lat[:, :].values))
wind_earth_1h_100m_strength_direction_solver = Eof(ddd, weights=weights_array)

with open('scratch/wind_earth/wind_earth_1h_100m_strength_direction_solver.pickle', 'wb') as f:
    pickle.dump(wind_earth_1h_100m_strength_direction_solver, f)

with open(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction_solver.pickle',
    'rb') as f:
    wind_earth_1h_100m_strength_direction_solver = pickle.load(f)


pcs = wind_earth_1h_100m_strength_direction_solver.pcs(npcs=10)
eofs = wind_earth_1h_100m_strength_direction_solver.eofs(neofs=10)
eigenvalues = wind_earth_1h_100m_strength_direction_solver.eigenvalues(
    neigs=10)
variance_fractions = \
    wind_earth_1h_100m_strength_direction_solver.varianceFraction(
    neigs=10)
# [0.39176041, 0.19642602, 0.1131851]
# endregion
# =============================================================================


# =============================================================================
# region visulization EOF

wind_earth_1h_100m_strength_direction = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')
lon = wind_earth_1h_100m_strength_direction.lon.values
lat = wind_earth_1h_100m_strength_direction.lat.values

with open(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction_solver.pickle',
    'rb') as f:
    wind_earth_1h_100m_strength_direction_solver = pickle.load(f)

# set colormap level and ticks
windlevel = np.arange(-32, 24.1, 0.4)
ticks = np.arange(-32, 24.1, 8)

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
    subplot_kw={'projection': transform})

plt_wind = ax.pcolormesh(
    lon, lat,
    wind_earth_1h_100m_strength_direction_solver.eofs()[2, :, :] * 10**4,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    rasterized=True)

cbar = fig.colorbar(plt_wind, orientation="horizontal", pad=0.1, fraction=0.09,
                    shrink=1, aspect=25, ticks=ticks, extend='both')
cbar.ax.set_xlabel("EOF3(0.113) of wind speed in 201001 [$10^{-4}$]")
# EOF1(0.392); EOF2(0.196); EOF3(0.113)
gl = ax.gridlines(crs=transform, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
ax.add_feature(borders); ax.add_feature(coastline)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

ax.set_extent(extent1km_lb, crs=transform)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig('figures/03_wind/3.2.2 wind speed in 201001_EOF3.png', dpi=1200)
plt.close('all')


'''
'''

# endregion
# =============================================================================




