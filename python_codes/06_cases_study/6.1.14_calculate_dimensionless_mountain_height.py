

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from windrose import WindroseAxes

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
    hm_m_model,
)


from DEoAI_analysis.module.statistics_calculate import(
    get_statistics,
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    inversion_layer,
)


# endregion
# =============================================================================


# =============================================================================
# region calculate dimensionless mountain height
istart = 0
ifinal = 217  # 217
filelist_ml = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
ml_3d_sim = xr.open_mfdataset(
    filelist_ml[istart:ifinal], concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override',
    chunks={'time': 1},
)
rlat = ml_3d_sim.rlat.values
rlon = ml_3d_sim.rlon.values
lat = ml_3d_sim.lat.values
lon = ml_3d_sim.lon.values
altitude = ml_3d_sim.altitude.values
time = ml_3d_sim.time.values
dim_height = np.zeros((len(time), lon.shape[0], lon.shape[1]))

strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})
wind_velocity = strength_direction_3d_20100801_09z.strength[
    istart:ifinal, 0:16, :, :].values


theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
theta = theta_3d_20100801_09z.theta[istart:ifinal, 0:16, :, :].values
# theta at h=1500
theta_hm = theta_3d_20100801_09z.theta[istart:ifinal, 15, :, :].values
# theta at h=10
theta_hsurf = theta_3d_20100801_09z.theta[istart:ifinal, 0, :, :].values

bruny_v_sqr = g/(np.mean(theta, axis=1)) * (theta_hm - theta_hsurf) / (1500-10)

h_dim = np.sqrt(bruny_v_sqr) * hm_m_model / np.mean(wind_velocity, axis=1)


h_dim = xr.Dataset(
    {"h_dim": (("time", "rlat", "rlon"), h_dim),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "rlat": rlat,
        "rlon": rlon,
    }
)
h_dim.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/h_dim_20100801_09.nc'
)



'''
######## check

h_dim = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/h_dim_20100801_09.nc'
)

istart = 30
filelist_ml = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
ml_3d_sim = xr.open_dataset(
    filelist_ml[istart]
)
strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})
wind_velocity = strength_direction_3d_20100801_09z.strength[
    istart, 0:16, :, :].values
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
theta = theta_3d_20100801_09z.theta[istart, 0:16, :, :].values

i_indices = 200

h_dim.h_dim[istart, i_indices, i_indices].values

np.sqrt(g/np.mean(theta[:, i_indices, i_indices]) * (
    theta[-1, i_indices, i_indices] - theta[0, i_indices, i_indices]
    )/(1500-10)) * hm_m_model / np.mean(wind_velocity[:, i_indices, i_indices])


'''

# endregion
# =============================================================================




