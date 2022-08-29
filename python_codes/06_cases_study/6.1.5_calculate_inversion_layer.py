

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


# region create ellipse mask

nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')

lon = nc3d_lb_c.lon.values
lat = nc3d_lb_c.lat.values

from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1


mask_e1 = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )
mask_e2 = ellipse(
    center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )
mask_e3 = ellipse(
    center_madeira[0] + radius_madeira[1] * 5 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    center_madeira[1] - radius_madeira[1] * 5 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )

'''
masked_e1 = np.ma.ones(lon.shape)
masked_e1[~mask_e1] = np.ma.masked
masked_e2 = np.ma.ones(lon.shape)
masked_e2[~mask_e2] = np.ma.masked
masked_e3 = np.ma.ones(lon.shape)
masked_e3[~mask_e3] = np.ma.masked

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,
)

ellipse1 = Ellipse(
    center_madeira,
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse1)

ellipse2 = Ellipse(
    [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
        center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse2)

ellipse3 = Ellipse(
    [center_madeira[0] + radius_madeira[1] * 5 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
        center_madeira[1] - radius_madeira[1] * 5 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse3)

model_lb1 = ax.contourf(
    lon, lat, masked_e1,
    transform=transform, colors='gainsboro', alpha=1, zorder=-3)
model_lb2 = ax.contourf(
    lon, lat, masked_e2,
    transform=transform, colors='gainsboro', alpha=1, zorder=-3)
model_lb3 = ax.contourf(
    lon, lat, masked_e3,
    transform=transform, colors='gainsboro', alpha=1, zorder=-3)

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
plt.savefig('figures/00_test/trial.png', dpi=300)

'''

# endregion
# =============================================================================


# =============================================================================
# region calculate inversion layer
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
inversion_base = np.zeros((len(time), lon.shape[0], lon.shape[1]))

for i in np.arange(0, len(time)):
    begin_time = datetime.datetime.now()
    
    tem = ml_3d_sim.T[i, :, :, :].values
    
    for j in np.arange(0, lon.shape[0]):  # 5):  #
        for k in np.arange(0, lon.shape[1]):  # 5):  #
            inversion_base[i, j, k] = inversion_layer(
                temperature=tem[:, j, k],
                altitude=altitude,
            )
    
    print(str(i) + '/' + str(len(time)) + "   " +
          str(datetime.datetime.now() - begin_time) + "   " +
          str(datetime.datetime.now()))

inversion_height = xr.Dataset(
    {"inversion_height": (("time", "rlat", "rlon"), inversion_base),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "rlat": rlat,
        "rlon": rlon,
    }
)
inversion_height.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc'
)



'''
######## check

inversion_height = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc'
)
istart = 30
filelist_ml = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
ml_3d_sim = xr.open_dataset(
    filelist_ml[istart]
)

i_indices = 200
inversion_height.inversion_height[istart, i_indices, i_indices].values
altitude = ml_3d_sim.altitude.values
temperature = ml_3d_sim.T[0, :, i_indices, i_indices].values


dinv = inversion_layer(temperature, altitude)
teminv = temperature[np.where(altitude == dinv)]
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(temperature, altitude, lw=0.5)
ax.scatter(teminv, dinv, s = 5)
fig.savefig('figures/00_test/trial.png')


'''

# endregion
# =============================================================================



