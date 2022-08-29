

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
# region interpolate Qc in z file

import timeit
import os
import numpy as np
import xarray as xr
from cdo import Cdo
cdo = Cdo()

# start = timeit.default_timer()


######## save model level in nc file
ds_const = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc'
)
model_lev = (ds_const.HHL[:,1:,:,:] + ds_const.HHL[:,:-1,:,:]) / 2.
model_lev_nc = xr.Dataset(
    {'ml': (['time', 'level1', 'rlat', 'rlon'], model_lev.values),
    #  "lat": (("rlat", "rlon"), ds_const['lat']),
    #  "lon": (("rlat", "rlon"), ds_const['lon']),
     },
    coords={'time': ('time', ds_const['time']),
            'rlat': ('rlat', ds_const['rlat']),
            'rlon': ('rlon', ds_const['rlon'])})
model_lev_nc.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/model_lev_nc.nc')
# model_lev_nc = xr.open_dataset(
#     'scratch/simulation/20100801_09_3d/02_lm_post_processed/model_lev_nc.nc')


######## save new z level in nc file
height_new = np.arange(100, 5100, 100)

nz = len(height_new)
nt, nmlp1, ny, nx = ds_const['HHL'].shape
z_lev = np.zeros((nt, nz, ny, nx))
for i in range(nz):
    z_lev[:, i, :, :] = height_new[i]
z_lev_nc = xr.Dataset(
    {'zlev': (['time', 'level1', 'rlat', 'rlon'], z_lev),
    #  "lat": (("rlat", "rlon"), ds_const['lat']),
    #  "lon": (("rlat", "rlon"), ds_const['lon']),
     },
    coords={'time': ('time', ds_const['time']),
            'rlat': ('rlat', ds_const['rlat']),
            'rlon': ('rlon', ds_const['rlon'])})
z_lev_nc.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/z_lev_nc.nc')
# z_lev_nc = xr.open_dataset(
#     'scratch/simulation/20100801_09_3d/02_lm_post_processed/z_lev_nc.nc')


######## save QC into new file

filelist = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd2010080*[0-9].nc'
))

ncfile = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
)
# ncfile.QC.shape

qc_model_lev = xr.Dataset(
    {'QC': (['time', 'level', 'rlat', 'rlon'], ncfile.QC),
    #  "lat": (("rlat", "rlon"), ncfile['lat']),
    #  "lon": (("rlat", "rlon"), ncfile['lon']),
     },
    coords={'time': ('time', ncfile['time']),
            'rlat': ('rlat', ncfile['rlat']),
            'rlon': ('rlon', ncfile['rlon'])})

qc_model_lev.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_model_lev.nc')
# qc_model_lev = xr.open_dataset(
#     'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_model_lev.nc')

# infile = 'scratch/simulation/20100801_09_3d/02_lm/lfsd20100801000000.nc'
# outfile = 'scratch/simulation/20100801_09_3d/02_lm_post_processed/lfsd20100801000000z.nc'

# Interpolate to new height levels
cdo.intlevel3d(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/model_lev_nc.nc',
    input='scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_model_lev.nc scratch/simulation/20100801_09_3d/02_lm_post_processed/z_lev_nc.nc',
    output='scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_z_lev.nc')

qc_z_lev = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_z_lev.nc')

qc_zlev_height = xr.Dataset(
    {'QC': (['time', 'height', 'rlat', 'rlon'], qc_z_lev.QC.values),
     "lat": (("rlat", "rlon"), ncfile['lat']),
     "lon": (("rlat", "rlon"), ncfile['lon']),
     },
    coords={'time': ('time', qc_z_lev['time']),
            'height': ('height', np.arange(100, 5100, 100)),
            'rlat': ('rlat', qc_z_lev['rlat']),
            'rlon': ('rlon', qc_z_lev['rlon'])})
qc_zlev_height.to_netcdf(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_zlev_height.nc')


'''
# check
ds_const = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc'
)
model_lev = (ds_const.HHL[:,1:,:,:] + ds_const.HHL[:,:-1,:,:]) / 2.
qc_model_lev = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_model_lev.nc')

qc_z_lev = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_z_lev.nc')
qc_z_lev.QC.shape
checkfile = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100803080000.nc'
)

# np.where(np.sum(checkfile.QC, axis = 1) == \
#     np.max(np.sum(checkfile.QC, axis = 1)))
# np.sum(checkfile.QC, axis = 1)[0, 798, 55]

checkfile.QC[0, :, 798, 55].values
model_lev[0, :, 798, 55].values
qc_z_lev.QC[56, :, 798, 55].values
qc_z_lev.zlev[56, :, 798, 55].values
# np.max(abs(qc_model_lev.QC[56, :, 798, 55].values - \
#     checkfile.QC[0, :, 798, 55].values))

import matplotlib.path as mpath
star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_T, = ax.plot(
    checkfile.QC[0, :, 798, 55].values,
    model_lev[0, :, 798, 55].values, '--r', marker=cut_star, markersize=2,
    linewidth=0.25, color='blue'
)
plt_T, = ax.plot(
    qc_z_lev.QC[56, :, 798, 55].values,
    qc_z_lev.zlev[56, :, 798, 55].values, '--b', marker=cut_star, markersize=2,
    linewidth=0.25, color='red'
)
plt.ylim(0, 5000)
# ax.set_xticks(np.arange(285, 316, 5))
# ax.set_yticks(np.arange(0, 3001, 500))
# ax.set_xticklabels(np.arange(285, 316, 5), size=8)
# ax.set_yticklabels(np.arange(0, 3.1, 0.5), size=8)
# ax.set_xlabel("Temperature in ERA5 [K]", size=10)
# ax.set_ylabel("Height [km]", size=10)

fig.subplots_adjust(left=0.14, right=0.95, bottom=0.2, top=0.99)
fig.savefig(
    'figures/00_test/trial.png', dpi=600)
plt.close('all')



'''

# endregion
# =============================================================================











