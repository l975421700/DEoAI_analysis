

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
# region create a bbox for era5 data

era5_pre_1979_2020 = xr.open_dataset(
    'scratch/obs/era5/monthly_pre_1979_2020.nc')
pre_lon = era5_pre_1979_2020.longitude.values
pre_lat = era5_pre_1979_2020.latitude.values
nc1h = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20051101000000.nc')


######## create a mask for Madeira
from matplotlib.path import Path
polygon = [
    (-23.401758, 31.897812), (-19.662523, 24.182158),
    (-11.290954, 26.82321), (-14.128447, 34.85296)
]
poly_path = Path(polygon)
x, y = np.meshgrid(pre_lon, pre_lat)
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
analysis_region_mask = poly_path.contains_points(coors).reshape(x.shape)


fig, ax = framework_plot(
    "1km", figsize=np.array([8.8, 9.2]) / 2.54,
)

ax.contourf(
    pre_lon, pre_lat, np.ones((len(pre_lat), len(pre_lon))),
    transform=transform, colors='gainsboro', alpha=0.25)
ax.contourf(
    nc1h.lon[80:920, 80:920], nc1h.lat[80:920, 80:920],
    np.ones(nc1h.lon[80:920, 80:920].shape),
    transform=transform, colors='gainsboro', alpha=0.5)
ax.scatter(
    [-23.401758, -19.662523, -14.128447, -11.290954],
    [31.897812, 24.182158, 34.85296, 26.82321],
    s=1)

mask_data = np.ma.array(np.ones(analysis_region_mask.shape))
mask_data.mask = (analysis_region_mask == False)
ax.pcolormesh(
    pre_lon, pre_lat, mask_data, edgecolors = 'none',
    transform=transform, alpha=0.25)

fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10.2.00_bbox for era5 data.png', dpi=300)


'''
nc1h.lon[80:920, 80:920][0, 0].values
nc1h.lon[80:920, 80:920][0, 839].values
nc1h.lon[80:920, 80:920][839, 0].values
nc1h.lon[80:920, 80:920][839, 839].values
nc1h.lat[80:920, 80:920][0, 0].values
nc1h.lat[80:920, 80:920][0, 839].values
nc1h.lat[80:920, 80:920][839, 0].values
nc1h.lat[80:920, 80:920][839, 839].values
(-23.401758, 31.897812); (-19.662523, 24.182158); (-14.128447, 34.85296); (-11.290954, 26.82321)
'''

# endregion
# =============================================================================




