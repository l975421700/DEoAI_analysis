

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
from metpy.calc.thermo import brunt_vaisala_frequency_squared
from haversine import haversine
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve

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
# region calculate 3d vorticity

filelist_ml = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
# ifinal = 2
ml_3d_sim = xr.open_mfdataset(
    filelist_ml, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
).metpy.parse_cf()

time = ml_3d_sim.time.values
rlon = ml_3d_sim.rlon.values
rlat = ml_3d_sim.rlat.values
lon = ml_3d_sim.lon.values
lat = ml_3d_sim.lat.values
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
zlev = ml_3d_sim.altitude.values
p0 = p0sl * np.exp(-(g * m * zlev / (r0 * t0sl)))


pollat = ml_3d_sim.rotated_pole.grid_north_pole_latitude
pollon = ml_3d_sim.rotated_pole.grid_north_pole_longitude
pollat_sin = np.sin(np.deg2rad(pollat))
pollat_cos = np.cos(np.deg2rad(pollat))
lon_rad = np.deg2rad(pollon - lon)
lat_rad = np.deg2rad(lat)
arg1 = pollat_cos * np.sin(lon_rad)
arg2 = pollat_sin * np.cos(lat_rad) - pollat_cos * \
    np.sin(lat_rad)*np.sin(lon_rad)
norm = 1.0/np.sqrt(arg1**2 + arg2**2)


# relative vorticity

rvorticity_3d_20100801_09z = xr.Dataset(
    {"relative_vorticity": (
        ("time", "zlev", "rlat", "rlon"),
        np.zeros((len(time), len(zlev), len(rlat), len(rlon)))),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "zlev": zlev,
        "rlat": rlat,
        "rlon": rlon,
    }
)

for i in np.arange(0, len(time)):
    begin_time = datetime.datetime.now()
    
    u = ml_3d_sim.U[i, :, :, :].values * units('m/s')
    v = ml_3d_sim.V[i, :, :, :].values * units('m/s')
    
    rvorticity_3d_20100801_09z.relative_vorticity[i, :, :, :] = \
        mpcalc.vorticity(
            u, v, dx[None, :], dy[None, :], dim_order='yx'
            )
    
    print(str(i) + ' / ' + str(len(time)) + "   " +
          str(datetime.datetime.now() - begin_time))

rvorticity_3d_20100801_09z.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/rvorticity_3d_20100801_09z.nc')

del rvorticity_3d_20100801_09z

# theta

theta_3d_20100801_09z = xr.Dataset(
    {"theta": (
        ("time", "zlev", "rlat", "rlon"),
        np.zeros((len(time), len(zlev), len(rlat), len(rlon)))),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "zlev": zlev,
        "rlat": rlat,
        "rlon": rlon,
    }
)

for i in np.arange(0, len(time)):
    begin_time = datetime.datetime.now()
    
    pp = ml_3d_sim.PP[i, :, :, :].values
    pressure = pp + p0[:, None, None]
    
    tem = ml_3d_sim.T[i, :, :, :].values
    theta_3d_20100801_09z.theta[
        i, :, :, :] = tem * (p0sl/pressure)**(r/cp)
    
    print(str(i) + ' / ' + str(len(time)) + "   " +
          str(datetime.datetime.now() - begin_time))

theta_3d_20100801_09z.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc')

del theta_3d_20100801_09z

# velocity

velocity_3d_20100801_09z = xr.Dataset(
    {"u_earth": (
        ("time", "zlev", "rlat", "rlon"),
        np.zeros((len(time), len(zlev), len(rlat), len(rlon)))),
     "v_earth": (
        ("time", "zlev", "rlat", "rlon"),
        np.zeros((len(time), len(zlev), len(rlat), len(rlon)))),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "zlev": zlev,
        "rlat": rlat,
        "rlon": rlon,
    }
)

for i in np.arange(0, len(time)):
    begin_time = datetime.datetime.now()
    
    u = ml_3d_sim.U[i, :, :, :].values * units('m/s')
    v = ml_3d_sim.V[i, :, :, :].values * units('m/s')
    
    velocity_3d_20100801_09z.u_earth[i, :, :, :] = \
        u * arg2 * norm + v * arg1 * norm
    velocity_3d_20100801_09z.v_earth[i, :, :, :] = \
        -u * arg1 * norm + v * arg2 * norm
    
    print(str(i) + ' / ' + str(len(time)) + "   " +
          str(datetime.datetime.now() - begin_time))

velocity_3d_20100801_09z.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')

del velocity_3d_20100801_09z
velocity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')

strength_direction_3d_20100801_09z = xr.Dataset(
    {"strength": (
        ("time", "zlev", "rlat", "rlon"),
        np.zeros((len(time), len(zlev), len(rlat), len(rlon)))),
     "direction": (
        ("time", "zlev", "rlat", "rlon"),
        np.zeros((len(time), len(zlev), len(rlat), len(rlon)))),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "zlev": zlev,
        "rlat": rlat,
        "rlon": rlon,
    }
)

for i in np.arange(0, len(time)):
    begin_time = datetime.datetime.now()
    
    u_earth = velocity_3d_20100801_09z.u_earth[
        i, :, :, :].values * units('m/s')
    v_earth = velocity_3d_20100801_09z.v_earth[
        i, :, :, :].values * units('m/s')
    
    strength_direction_3d_20100801_09z.strength[i, :, :, :] = \
        (u_earth.magnitude**2 + v_earth.magnitude**2)**0.5
    strength_direction_3d_20100801_09z.direction[i, :, :, :] = \
        mpcalc.wind_direction(
            u=u_earth,
            v=v_earth,
            convention='to')
    
    print(str(i) + ' / ' + str(len(time)) + "   " +
          str(datetime.datetime.now() - begin_time))

strength_direction_3d_20100801_09z.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc')


'''
# check
rvorticity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/rvorticity_3d_20100801_09z.nc')
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc')
velocity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')
strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc')


icheck = 200
izlev = 50

filelist_ml = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
ml_3d_sim = xr.open_dataset(filelist_ml[icheck])
lon = ml_3d_sim.lon.values
lat = ml_3d_sim.lat.values
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
zlev = ml_3d_sim.altitude.values
p0 = p0sl * np.exp(-(g * m * zlev / (r0 * t0sl)))
time = ml_3d_sim.time.values
pp = ml_3d_sim.PP.values
tem = ml_3d_sim.T.values
pollat = ml_3d_sim.rotated_pole.grid_north_pole_latitude
pollon = ml_3d_sim.rotated_pole.grid_north_pole_longitude

# vorticity

u = ml_3d_sim.U[0, izlev, :, :].values.squeeze() * units('m/s')
v = ml_3d_sim.V[0, izlev, :, :].values.squeeze() * units('m/s')

rvor = mpcalc.vorticity(u, v, dx, dy, dim_order='yx')

np.max(abs(rvorticity_3d_20100801_09z.relative_vorticity[icheck, izlev, :, :].values - rvor.magnitude))

# theta
pressure = pp[0, izlev, :, :] + p0[izlev]
theta = ml_3d_sim.T.values[0, izlev, :, :] * (p0sl/pressure)**(r/cp)

np.max(abs(theta_3d_20100801_09z.theta[icheck, izlev, :, :].values - theta))

# u earth and v earth

u_earth, v_earth = rotate_wind(u, v, lat, lon, pollat, pollon)
np.max(abs(velocity_3d_20100801_09z.u_earth[icheck, izlev, :, :].values - u_earth.magnitude))
np.max(abs(velocity_3d_20100801_09z.v_earth[icheck, izlev, :, :].values - v_earth.magnitude))

# strength/direction
np.max(abs(strength_direction_3d_20100801_09z.strength[icheck, izlev, :, :].values - \
    (u_earth.magnitude**2 + v_earth.magnitude**2)**0.5))



'''



# endregion
# =============================================================================





