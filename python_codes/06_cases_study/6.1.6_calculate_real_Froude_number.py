

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
# region calculate real froude number

################################ wind velocity
velocity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')
u_earth = velocity_3d_20100801_09z.u_earth[
    68:200, 0:17, 698:(698 + 60), 520:(520 + 60)].astype('float16').values
v_earth = velocity_3d_20100801_09z.v_earth[
    68:200, 0:17, 698:(698 + 60), 520:(520 + 60)].astype('float16').values
zlev = velocity_3d_20100801_09z.zlev[0:17].values
wind_strength = (u_earth**2 + v_earth**2)**0.5
time = velocity_3d_20100801_09z.time.values[68:200]

################################ inversion base
inversion_height_20100801_09 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc')
inversion_height = inversion_height_20100801_09.inversion_height[
    68:200, 698:(698 + 60), 520:(520 + 60)].astype('float16').values


rlat = inversion_height_20100801_09.rlat[698:(698 + 60)].values
rlon = inversion_height_20100801_09.rlon[520:(520 + 60)].values
lon = inversion_height_20100801_09.lon[698:(698 + 60), 520:(520 + 60)].values
lat = inversion_height_20100801_09.lat[698:(698 + 60), 520:(520 + 60)].values

def froude_number(velocity, zlev, height, g):
    '''
    Input --------
    velocity: vertically averaged velocity
    zlevel: altitude corresponding to velocity
    height: inversion height
    g: gravity acceleration, 9.80665
    
    Output --------
    fr: froude number
    
    Example --------
    velocity = wind_strength[0, :, 0, 0]
    zlev = zlev
    height = inversion_height[0, 0, 0]
    g = g
    
    '''
    
    v = velocity[np.where(zlev <= height)[0]].mean()
    fr = v / (g * height)**0.5
    
    return(fr)


fr = np.zeros_like(inversion_height, dtype='float64')

for i in range(fr.shape[0]):
    for j in range(fr.shape[1]):
        for k in range(fr.shape[2]):
            fr[i, j, k] = froude_number(
                velocity=wind_strength[i, :, j, k],
                zlev=zlev,
                height=inversion_height[i, j, k],
                g = g
            )
    print(str(i) + '/' + str(fr.shape[0]))


froude_number_2010080320_0907 = xr.Dataset(
    {"fr": (("time", "rlat", "rlon"), fr),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "rlat": rlat,
        "rlon": rlon,
    }
)

froude_number_2010080320_0907.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/froude_number_2010080320_0907.nc'
)


'''
######## check
i = 20
j = 40
k = 40

velocity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')
u_earth = velocity_3d_20100801_09z.u_earth[
    68 + i, 0:17, 698 + j, 520 + k].astype('float16').values
v_earth = velocity_3d_20100801_09z.v_earth[
    68 + i, 0:17, 698 + j, 520 + k].astype('float16').values
zlev = velocity_3d_20100801_09z.zlev[0:17].values
wind_strength = (u_earth**2 + v_earth**2)**0.5
time = velocity_3d_20100801_09z.time.values[68 + i]

inversion_height_20100801_09 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc')
inversion_height = inversion_height_20100801_09.inversion_height[
    68 + i, 698 + j, 520 + k].astype('float16').values

velocity = wind_strength
height = inversion_height
g = g
v = velocity[np.where(zlev <= height)[0]].mean()
fr = v / (g * height)**0.5

froude_number_2010080320_0907 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/froude_number_2010080320_0907.nc'
)
froude_number_2010080320_0907.fr[i, j, k].values
fr
'''
# endregion
# =============================================================================



