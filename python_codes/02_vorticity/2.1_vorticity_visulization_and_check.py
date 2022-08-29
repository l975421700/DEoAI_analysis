

# =============================================================================
# region import packages

import datetime
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font',
       family='Times New Roman', size=10)

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
import rasterio as rio

# import self defined functions # print(sys.path)
import sys
sys.path.append('/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
from DEoAI_analysis.module.mapplot import ticks_labels, scale_bar
from DEoAI_analysis.module.mapplot import framework_plot

import os
import re
import glob

# endregion


# =============================================================================
# region visualize vorticity

relative_vorticity_1km_1h201002 = xr.open_dataset(
    "scratch/relative_vorticity_1km_1h/relative_vorticity_1km_1h201002.nc")

fig, ax = framework_plot(
    "1km", figsize = np.array([8.8, 7.5]) / 2.54, country_boundaries = False
)

transform = ccrs.PlateCarree()
hours = 224
# 216:228
vorlevel = np.arange(-20, 20.1, 2)
plt_r_vorticity = ax.contourf(
    relative_vorticity_1km_1h201002.lon,
    relative_vorticity_1km_1h201002.lat,
    relative_vorticity_1km_1h201002.relative_vorticity[hours, :, :] * 10**4,
    levels = vorlevel[abs(vorlevel) >= 6],
    transform=transform, cmap='bwr', extend='both')

cbar = fig.colorbar(
    plt_r_vorticity, orientation="horizontal",  pad=0.12, fraction=0.12,
    shrink=0.8, aspect=25)
cbar.ax.set_xlabel("Relative vorticity * $10^{-4}$ ($s^{-1}$)")


coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.5)
ax.add_feature(coastline)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
    facecolor='none', lw=0.5)
ax.add_feature(borders)

fig.subplots_adjust(left=0.2, right=0.99, bottom=0.2, top=0.99)
fig.savefig('figures/test/trial.png', dpi = 300)


# endregion


# =============================================================================
# region check vorticity calculation
relative_vorticity_1km_1h201001 = xr.open_dataset(
    "scratch/relative_vorticity_1km_1h/relative_vorticity_1km_1h201001.nc")

lffd2010011010 = xr.open_dataset("/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20100110100000.nc")

lffd2010011010 = lffd2010011010.metpy.parse_cf()
lons = lffd2010011010.lon.data
lats = lffd2010011010.lat.data

u_10m = lffd2010011010.U_10M.squeeze().values * units('m/s')
v_10m = lffd2010011010.V_10M.squeeze().values * units('m/s')

dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Calculation based on mpcalc
avor = mpcalc.absolute_vorticity(
    u_10m, v_10m,
    dx, dy, lats * units.degrees, dim_order='yx')
fcoriolis = mpcalc.coriolis_parameter(np.deg2rad(lats))
rvor = mpcalc.vorticity(u_10m, v_10m, dx, dy, dim_order='yx')
np.max(abs(rvor.magnitude -
           relative_vorticity_1km_1h201001.relative_vorticity[226, :, :]))


# endregion

# =============================================================================
# region check vorticity calculation

relative_vorticity_1km_1h_100m201001 = xr.open_dataset(
    "/project/pr94/qgao/DEoAI/scratch/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201001.nc")

lffd2010011010 = xr.open_dataset(
    "/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_100m/lffd20100110100000z.nc")

lffd2010011010 = lffd2010011010.metpy.parse_cf()
lons = lffd2010011010.lon.data
lats = lffd2010011010.lat.data

u_100m = lffd2010011010.U.squeeze().values * units('m/s')
v_100m = lffd2010011010.V.squeeze().values * units('m/s')

dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

rvor = mpcalc.vorticity(
    u_100m,
    v_100m,
    dx, dy, dim_order='yx')

np.max(abs(rvor.magnitude -
           relative_vorticity_1km_1h_100m201001.relative_vorticity[226, :, :]))

# endregion


