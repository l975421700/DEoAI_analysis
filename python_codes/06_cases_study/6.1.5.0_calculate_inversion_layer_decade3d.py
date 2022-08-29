

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
# region calculate inversion layer in decadal 3d Tenerif

ncfile_constant = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd20051101000000c.nc')
rlon = ncfile_constant.rlon.values
rlat = ncfile_constant.rlat.values
lon = ncfile_constant.lon.values
lat = ncfile_constant.lat.values
hsurf = ncfile_constant.HSURF.values

for i in np.arange(110, 120):  # range(len(years_months)):
    # i=0
    filelist = np.array(sorted(glob.glob(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd20' + years_months[i] + '*[0-9].nc')))
    ncfiles = xr.open_mfdataset(
        filelist, concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override')
    time = ncfiles.time.values
    altitude = ((ncfiles.vcoord[:-1] + ncfiles.vcoord[1:])/2).values
    inversion_base = np.zeros((len(time), lon.shape[0], lon.shape[1]))
    
    for j in range(len(time)):
        # j = 0
        begin_time = datetime.datetime.now()
        print(begin_time)
        
        tem = ncfiles.T[j].values
        for k in range(lon.shape[0]):
            # k=0
            for l in range(lon.shape[1]):
                # l=0
                inversion_base[j, k, l] = inversion_layer(
                    temperature=tem[::-1, k, l],
                    altitude=altitude[::-1],
                    topo=hsurf[0, k, l])
        print(
            str(i) + '/' + str(len(years_months)) + ' ' + \
            str(j) + '/' + str(len(time)) + ' ' + \
            str(datetime.datetime.now() - begin_time))
    
    inversion_height_tenerif3d = xr.Dataset(
        {"inversion_height": (("time", "rlat", "rlon"), inversion_base),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon), },
        coords={
            "time": time,
            "rlat": rlat,
            "rlon": rlon, })
    inversion_height_tenerif3d.to_netcdf(
        'scratch/inversion_height/tenerif3d/' +
        'inversion_height_tenerif3d_20' + years_months[i] + '.nc')


'''
######## check
# find nearest grid to tenerif
nc3D_Tenerife_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd20051101000000c.nc')
from DEoAI_analysis.module.spatial_analysis import find_nearest_grid
tenerif_loc = np.array([28.3183, -16.3822])
nearestgrid_indices = find_nearest_grid(
    tenerif_loc[0], tenerif_loc[1],
    nc3D_Tenerife_c.lat.values,
    nc3D_Tenerife_c.lon.values)
(nc3D_Tenerife_c.lat.values[nearestgrid_indices],
 nc3D_Tenerife_c.lon.values[nearestgrid_indices],)
nc3D_Tenerife_c.HSURF[0, 29, 55].values
(157.17213)
i=0
j = 240
k = 29
l = 55

inversion_height_tenerif3d = xr.open_dataset(
    'scratch/inversion_height/tenerif3d/' + \
        'inversion_height_tenerif3d_20' + years_months[i] + '.nc')
filelist = np.array(sorted(glob.glob(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd20' + years_months[i] + '*[0-9].nc')))
ncfile = xr.open_dataset(filelist[j])
altitude = ((ncfile.vcoord[:-1] + ncfile.vcoord[1:])/2).values

tem = ncfile.T[0, :, k, l].values

inversion_height_tenerif3d.inversion_height[j, k, l].values
inversion_layer(temperature=tem[::-1], altitude=altitude[::-1],)
inversion_layer(
    temperature=tem[::-1], altitude=altitude[::-1],
    topo = nc3D_Tenerife_c.HSURF[0, 29, 55].values
)

dinv = inversion_layer(tem[::-1], altitude[::-1])
teminv = tem[::-1][np.where(altitude[::-1] == dinv)]

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(tem[::-1], altitude[::-1], '.-', color='gray', lw=0.5, markersize=2.5)
ax.scatter(teminv, dinv, s=5, c='blue', zorder=10)

ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xticks(np.arange(245, 305.1, 10))
ax.set_ylim(0, 5000)
ax.set_xlim(245, 305)
ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)

fig.savefig('figures/00_test/trial1.png')
'''
# endregion
# =============================================================================


# =============================================================================
# region calculate inversion layer in decadal 3d Madeira

ncfile_constant = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20100809230000.nc')
rlon = ncfile_constant.rlon.values
rlat = ncfile_constant.rlat.values
lon = ncfile_constant.lon.values
lat = ncfile_constant.lat.values

for i in np.arange(110, 120):  # range(len(years_months)):
    # i=0
    filelist = np.array(sorted(glob.glob(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20' + years_months[i] + '*[0-9].nc')))
    ncfiles = xr.open_mfdataset(
        filelist, concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override')
    time = ncfiles.time.values
    altitude = ((ncfiles.vcoord[:-1] + ncfiles.vcoord[1:])/2).values
    inversion_base = np.zeros((len(time), lon.shape[0], lon.shape[1]))
    
    for j in range(len(time)):
        # j = 0
        begin_time = datetime.datetime.now()
        print(begin_time)
        
        tem = ncfiles.T[j].values
        for k in range(lon.shape[0]):
            # k=0
            for l in range(lon.shape[1]):
                # l=0
                inversion_base[j, k, l] = inversion_layer(
                    temperature=tem[::-1, k, l],
                    altitude=altitude[::-1],)
        print(
            str(i) + '/' + str(len(years_months)) + ' ' + \
            str(j) + '/' + str(len(time)) + ' ' + \
            str(datetime.datetime.now() - begin_time))
    
    inversion_height_madeira3d = xr.Dataset(
        {"inversion_height": (("time", "rlat", "rlon"), inversion_base),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon), },
        coords={
            "time": time,
            "rlat": rlat,
            "rlon": rlon, })
    inversion_height_madeira3d.to_netcdf(
        'scratch/inversion_height/madeira3d/' + \
        'inversion_height_madeira3d_20' + years_months[i] + '.nc')


'''
######## check
i=0
j = 732
k = 4
l = 25

inversion_height_madeira3d = xr.open_dataset(
    'scratch/inversion_height/madeira3d/' + \
        'inversion_height_madeira3d_20' + years_months[i] + '.nc')
filelist = np.array(sorted(glob.glob(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20' + years_months[i] + '*[0-9].nc')))
ncfile = xr.open_dataset(filelist[j])
altitude = ((ncfile.vcoord[:-1] + ncfile.vcoord[1:])/2).values

tem = ncfile.T[0, :, k, l].values

inversion_height_madeira3d.inversion_height[j, k, l].values
inversion_layer(temperature=tem[::-1], altitude=altitude[::-1],)

dinv = inversion_layer(tem[::-1], altitude[::-1])
teminv = tem[::-1][np.where(altitude[::-1] == dinv)]

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(tem[::-1], altitude[::-1], '.-', color='gray', lw=0.5, markersize=2.5)
ax.scatter(teminv, dinv, s=5, c='blue', zorder=10)

ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xticks(np.arange(245, 305.1, 10))
ax.set_ylim(0, 5000)
ax.set_xlim(245, 305)
ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)

fig.savefig('figures/00_test/trial1.png')
'''
# endregion
# =============================================================================


