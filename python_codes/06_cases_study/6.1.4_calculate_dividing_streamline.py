

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
from time import sleep

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
from matplotlib.patches import Ellipse

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

from joblib import Memory


# endregion
# =============================================================================


# =============================================================================
# region find indices for ellipse 3

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

(min(np.where(mask_e3)[0]), max(np.where(mask_e3)[0]), min(np.where(mask_e3)[1]), max(np.where(mask_e3)[1]),)

(min(np.where(mask_e2)[0]), max(np.where(mask_e2)[0]), min(np.where(mask_e2)[1]), max(np.where(mask_e2)[1]),)

# endregion
# =============================================================================


istart = 139  # 68
ifinal = 200  # 200

# =============================================================================
# region calculate dividing stramline of madeira

nc3D_Madeira_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc')
hm_m_model = np.max(nc3D_Madeira_c.HHL[0, -1, :, :].values)

nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
upstream_m_rlat = np.arange(639, 670)
upstream_m_rlon = np.arange(620, 662)
rlat = nc3d_lb_c.rlat[639:670].values
rlon = nc3d_lb_c.rlon[620:662].values
lon = nc3d_lb_c.lon[639:670, 620:662].values
lat = nc3d_lb_c.lat[639:670, 620:662].values


# (639, 670, 620, 662) has (640, 668, 621, 660)
# (698, 698 + 60, 520, 520 + 60) = (698, 758, 520, 580) has (725, 754, 536, 575)
# filelist_ml = sorted(glob.glob(
#     'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
# ml_3d_sim_all = xr.open_mfdataset(
#     filelist_ml[istart:ifinal], concat_dim="time",
#     data_vars='minimal', coords='minimal', compat='override',
#     chunks = {'time': 1},
# )
# zlev = ml_3d_sim_all.altitude[0:17].values
# time = ml_3d_sim_all.time.values
del nc3D_Madeira_c, nc3d_lb_c

def dividing_streamline(ds_height, ds_theta, ds_u, hm, hc_guess):
    '''
    Input --------
    ds_height: height levels
    ds_theta: potential temperature at height levels
    ds_u: velocity at height levels
    hm: maximum height of mountain
    hc_guess: initial guess of the height of dividing streamline
    
    Output --------
    ds: dividing stream line
    
    '''
    
    brunt_v_sqr = brunt_vaisala_frequency_squared(
        ds_height * units.meter, ds_theta * units.kelvin)
    brunt_v_sqr_f = interpolate.interp1d(
        ds_height, brunt_v_sqr.magnitude, fill_value="extrapolate")
    
    velocity_sqr_f = interpolate.interp1d(
        ds_height, ds_u**2, fill_value="extrapolate")
    
    def integrand(z):
        return brunt_v_sqr_f(z)*(hm - z)
    
    def func(hc):
        y, err = quad(
            integrand,
            hc, hm, epsabs=10**-4, epsrel=10**-4, limit=50)
        y = y - velocity_sqr_f(hc)/2
        return y
    
    if hc_guess < hm*0.2:
        hc_guess = hm*0.6
    if hc_guess > hm:
        hc_guess = hm * 0.6
    
    hc = fsolve(func, hc_guess, col_deriv=True)
    # hc1 = bisect(func, 10, 1500)
    
    return hc

hc_guess = hm_m_model*0.6

for i in np.arange(istart, ifinal):
    # i = istart
    theta_3d_20100801_09z = xr.open_dataset(
        'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
    strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})
    
    zlev = theta_3d_20100801_09z.zlev[0:17].values
    time = theta_3d_20100801_09z.time.values
    
    theta = theta_3d_20100801_09z.theta[
        i, 0:17, upstream_m_rlat, upstream_m_rlon].values
    velocity = strength_direction_3d_20100801_09z.strength[
        i, 0:17, upstream_m_rlat, upstream_m_rlon].values
    
    ds_hc = np.zeros((1, len(upstream_m_rlat), len(upstream_m_rlon)))
    
    for j in np.arange(0, len(upstream_m_rlat)):  # 5):  #
        for k in np.arange(0, len(upstream_m_rlon)):  # 5):  #
            begin_time = datetime.datetime.now()
            
            ds_hc[0, j, k] = dividing_streamline(
                ds_height=zlev,
                ds_theta=theta[:, j, k],
                ds_u=velocity[:, j, k],
                hm=hm_m_model,
                hc_guess=hc_guess
            )
            
            hc_guess = ds_hc[0, j, k]
            
            print(str(i) + '/' + str(ifinal-1) + "   " +
                  str(j) + '/' + str(len(upstream_m_rlat)) + "   " +
                  str(k) + '/' + str(len(upstream_m_rlon)) + "   " +
                  str(datetime.datetime.now() - begin_time) + "   " +
                  str(datetime.datetime.now()))
    
    dividing_streamline_2010080320_0907 = xr.Dataset(
        {"ds_height": (("time", "rlat", "rlon"), ds_hc),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon),
         },
        coords={
            "time": [time[i]],
            "rlat": rlat,
            "rlon": rlon,
        }
    )
    dividing_streamline_2010080320_0907.to_netcdf(
        'scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline/dividing_streamline_201008' + str(time[i])[8:10] + '_' + str(time[i])[11:13] + '.nc'
    )
    
    del theta, velocity, theta_3d_20100801_09z, ds_hc, strength_direction_3d_20100801_09z, dividing_streamline_2010080320_0907



'''
# https://stackoverflow.com/questions/27382952/python-fsolve-with-unknown-inside-the-upper-limit-of-an-integral

######## check
nc3D_Madeira_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc')
hm_m_model = np.max(nc3D_Madeira_c.HHL[0, -1, :, :].values)
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})


i = 0

filelist_ds = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline/dividing_streamline_201008*.nc'))

dividing_streamline_2010080320_0907 = xr.open_dataset(
    filelist_ds[i]
)

filelist_ml = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*z.nc'))
ml_3d_sim_all = xr.open_dataset(
    filelist_ml[i+68]
)

indices = 0

ds_height = ml_3d_sim_all.altitude[0:17].values
ds_theta = theta_3d_20100801_09z.theta[
    i+68, 0:17, 698 + indices, 520 + indices].values
hm = hm_m_model
ds_u = strength_direction_3d_20100801_09z.strength[
    i+68, 0:17, 698 + indices, 520 + indices].values
dividing_streamline(ds_height, ds_theta, ds_u, hm, 1400)

dividing_streamline_2010080320_0907.ds_height[0, indices, indices].values

# np.min(dividing_streamline_2010080320_0907.ds_height)
# np.sum(dividing_streamline_2010080320_0907.ds_height.values > hm)


######## test
i = 0
ds_height = zlev
ds_theta = theta_3d_20100801_09z.theta[i, :, 698, 520].values
hm = hm_m_model
ds_u = strength_direction_3d_20100801_09z.strength[i, :, 698, 520].values
dividing_streamline(ds_height, ds_theta, ds_u, hm, hc_guess)

j = 0
k=0
ds_height=zlev
ds_theta=theta[:, j, k]
ds_u=velocity[:, j, k]
hm=hm_m_model
hc_guess=hm_m_model*0.6
dividing_streamline(ds_height, ds_theta, ds_u, hm, hc_guess)


# profile
from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(dividing_streamline)
lp_wrapper(ds_height, ds_theta, ds_u, hm, hc_guess)
lp.print_stats()

# %time dividing_streamline(ds_height, ds_theta, ds_u, hm, hc_guess)

# interpolate.interp1d
xnew = np.arange(ds_height[0], ds_height[-1], 10)
ynew = brunt_v_sqr_f(xnew)   # use interpolation function returned by `interp1d`
fig, ax = plt.subplots()
ax.plot(
    ds_height,
    brunt_v_sqr.magnitude,
    linewidth=3, color='blue',
)
ax.plot(
    xnew,
    ynew,
    linewidth=1, color='red',
)
ax.set(xlabel='bv', ylabel='height')
fig.savefig('figures/00_test/trial.png', dpi=300)

# check difference
dividing_streamline1 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline_2010080320.nc')
dividing_streamline2 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline/dividing_streamline_20100803_20.nc')

np.mean(abs(dividing_streamline1.ds_height.values -
       dividing_streamline2.ds_height.values))


'''

# endregion
# =============================================================================


# =============================================================================
# region spatial plot of dividing stramline

nc3D_Madeira_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc')
hm_m_model = np.max(nc3D_Madeira_c.HHL[0, -1, :, :]).values

nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
hsurf = nc3d_lb_c.HHL[0, -1, 698:(698 + 60), 520:(520 + 60)].values

dividing_streamline_2010080320 = xr.open_dataset(
    '/project/pr94/qgao/DEoAI/scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline/dividing_streamline_20100809_07.nc')

# np.sum(dividing_streamline_2010080320.ds_height.values[0, :, :] > hsurf)

ds_height = dividing_streamline_2010080320.ds_height[0, :, :].values
ds_height[ds_height > hm_m_model] = hm_m_model
np.sum(dividing_streamline_2010080320.ds_height[0, :, :].values > hm_m_model)
np.sum(ds_height > hm_m_model)

froude_n = 1 - ds_height/hm_m_model



extentm_up = [
    np.min(dividing_streamline_2010080320.lon.values),
    np.max(dividing_streamline_2010080320.lon.values),
    np.min(dividing_streamline_2010080320.lat.values),
    np.max(dividing_streamline_2010080320.lat.values),
    ]
ticklabelm_up = ticks_labels(
    np.min(dividing_streamline_2010080320.lon.values),
    np.max(dividing_streamline_2010080320.lon.values),
    np.min(dividing_streamline_2010080320.lat.values),
    np.max(dividing_streamline_2010080320.lat.values),
    0.2, 0.2
    )

fig, ax = framework_plot(
    "self_defined", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=8,
    extent=extentm_up, ticklabel=ticklabelm_up
)
demlevel = np.arange(0, 0.301, 0.001)
ticks = np.arange(0, 0.301, 0.05)
plt_dem = ax.pcolormesh(
    dividing_streamline_2010080320.lon, dividing_streamline_2010080320.lat,
    froude_n,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('RdBu', len(demlevel)), rasterized=True,
    transform=transform,
)
cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.12, fraction=0.12,
    shrink=0.8, aspect=25, ticks=ticks, extend='neither')
cbar.ax.set_xlabel("Froude number at 20100803 20:00 UTC [-]")

# demlevel = np.arange(1094, 1550.1, 1)
# ticks = np.arange(1100, 1501, 100)
# plt_dem = ax.pcolormesh(
#     dividing_streamline_2010080320.lon, dividing_streamline_2010080320.lat,
#     ds_height,
#     norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
#     cmap=cm.get_cmap('RdBu', len(demlevel)), rasterized=True,
#     transform=transform,
# )
# cbar = fig.colorbar(
#     plt_dem, orientation="horizontal",  pad=0.12, fraction=0.12,
#     shrink=0.8, aspect=25, ticks=ticks, extend='neither')
# cbar.ax.set_xlabel("Height of dividing streamline [m]")

# plot ellipse
# ellipse2 = Ellipse(
#     [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
#         np.deg2rad(angle_deg_madeira)),
#         center_madeira[1] + radius_madeira[0] * 3 * np.sin(
#         np.deg2rad(angle_deg_madeira))],
#     radius_madeira[0] * 2, radius_madeira[1] * 2,
#     angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
# ax.add_patch(ellipse2)
ellipse3 = Ellipse(
    [center_madeira[0] + radius_madeira[1] * 5 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
        center_madeira[1] - radius_madeira[1] * 5 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse3)

scale_bar(ax, bars=2, length=20, location=(0.05, 0.025),
          barheight=1, linewidth=0.2, col='black')

ax.set_extent(extentm_up, crs=transform)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.1, top=0.98)
fig.savefig('figures/00_test/trial2.png', dpi=600)







# endregion
# =============================================================================




