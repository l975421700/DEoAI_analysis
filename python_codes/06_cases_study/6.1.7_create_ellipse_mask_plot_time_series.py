

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
from scipy import stats

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

from DEoAI_analysis.module.vortex_namelist import (
    correctly_identified, correctly_reidentified
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics,
)

from DEoAI_analysis.module.spatial_analysis import(
    find_nearest_grid,
    rotate_wind,
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

# ellipse1 = Ellipse(
#     center_madeira,
#     radius_madeira[0] * 2, radius_madeira[1] * 2,
#     angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
# ax.add_patch(ellipse1)

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
# region windrose in ellipse

wind_earth_1h_100m = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc',
    chunks={"time": 5})

istart = 5156  # 5156
ifinal = 5288  # 5288

wind_strength = wind_earth_1h_100m.strength[istart:ifinal, :, :].values
wind_direction = wind_earth_1h_100m.direction[istart:ifinal, :, :].values

wind_strength_e1 = wind_strength[:, mask_e1].flatten()
wind_direction_e1 = wind_direction[:, mask_e1].flatten()
wind_strength_e2 = wind_strength[:, mask_e2].flatten()
wind_direction_e2 = wind_direction[:, mask_e2].flatten()
wind_strength_e3 = wind_strength[:, mask_e3].flatten()
wind_direction_e3 = wind_direction[:, mask_e3].flatten()


# plot ellipse e1
windbins = np.arange(6, 13.1, 1, dtype='int32')

fig, ax = plt.subplots(figsize=np.array([8.8, 7]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=1200)
ax.set_extent([0, 1, 0, 1])
windrose_ax = inset_axes(
    ax, width=2.2, height=2.2, loc=10, bbox_to_anchor=(0.45, 0.5),
    bbox_transform=ax.transData, axes_class=WindroseAxes
    )
windrose_ax.bar(
    wind_direction_e3, wind_strength_e3, normed=True,
    opening=1, edgecolor=None, nsector=72,
    bins=windbins,
    cmap=cm.get_cmap('RdBu_r', len(windbins - 1)),
    label='Wind velocity [m/s]',
    )

windrose_ax._info['bins'][-1] = 14

windrose_legend = windrose_ax.legend(
    loc=(1.05, 0.15),
    decimal_places=0, ncol=1,
    borderpad=0.1,
    labelspacing=0.5, handlelength=1.2, handletextpad=0.6,
    fancybox=False,
    fontsize=8,
    frameon=False,
    title='Wind velocity [m/s]', title_fontsize=8,
    )
windrose_ax.grid(alpha=0.5, ls='--', lw=0.5)

for lh in windrose_legend.legendHandles:
    lh.set_edgecolor(None)
windrose_ax.tick_params(axis='x', which='major', pad=0)

windrose_ax.set_yticks(np.arange(5, 25.1, step=5))
windrose_ax.set_yticklabels([5, 10, 15, 20, '25%'])

ax.axis('off')
# ax.text(0.5, 0.5, 'Ellipse $e_3$', zorder = 2)
fig.subplots_adjust(left=0.01, right=0.8, bottom=0.2, top=0.8)
fig.savefig('figures/03_wind/3.5.4 wind rose in 20100803_09_e3.png', dpi=1200)
# fig.savefig('figures/00_test/trial.png', dpi=300)


# plot ellipse e2
windbins = np.array((0, 3, 4, 5, 6, 7, 8), dtype='int32')

fig, ax = plt.subplots(figsize=np.array([8.8, 7]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=1200)
ax.set_extent([0, 1, 0, 1])
windrose_ax = inset_axes(
    ax, width=2.2, height=2.2, loc=10, bbox_to_anchor=(0.45, 0.5),
    bbox_transform=ax.transData, axes_class=WindroseAxes
)
windrose_ax.bar(
    wind_direction_e2, wind_strength_e2, normed=True,
    opening=1, edgecolor=None, nsector=72,
    bins=windbins,
    cmap=cm.get_cmap('RdBu_r', len(windbins - 1)),
    label='Wind velocity [m/s]',
)

windrose_ax._info['bins'][-1] = 10

windrose_legend = windrose_ax.legend(
    loc=(1.05, 0.15),
    decimal_places=0, ncol=1,
    borderpad=0.1,
    labelspacing=0.5, handlelength=1.2, handletextpad=0.6,
    fancybox=False,
    fontsize=8,
    frameon=False,
    title='Wind velocity [m/s]', title_fontsize=8,
)
windrose_ax.grid(alpha=0.5, ls='--', lw=0.5)

for lh in windrose_legend.legendHandles:
    lh.set_edgecolor(None)
windrose_ax.tick_params(axis='x', which='major', pad=0)

windrose_ax.set_yticks(np.arange(2, 8.1, step=2))
windrose_ax.set_yticklabels([2, 4, 6, '8%'])

ax.axis('off')
# ax.text(0.1, 0.1, 'Mean wind direction: 79° + 180°')
fig.subplots_adjust(left=0.01, right=0.8, bottom=0.2, top=0.8)
fig.savefig('figures/03_wind/3.5.5 wind rose in 20100803_09_e2.png', dpi=1200)
# fig.savefig('figures/00_test/trial.png', dpi=300)

'''
270 - (np.mean(wind_direction) - 180) - 180
'''

# endregion
# =============================================================================


# =============================================================================
# region time series during case study


################################ dividing streamline
ds_filelist = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline/dividing_streamline_201008*.nc'))
dividing_streamline_201008 = xr.open_mfdataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/dividing_streamline/dividing_streamline_201008*.nc',
    concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

time = dividing_streamline_201008.time.values
lon = dividing_streamline_201008.lon.values
lat = dividing_streamline_201008.lat.values
ds_height = dividing_streamline_201008.ds_height.values

ds_mask_e3 = ellipse(
    center_madeira[0] + radius_madeira[1] * 5 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    center_madeira[1] - radius_madeira[1] * 5 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
)

# froude_n = 1 - ds_height[0, :, :]/hm_m_model
mean_ds_height = np.ones((len(time), np.sum(ds_mask_e3)))
for i in np.arange(0, len(time)):
    mean_ds_height[i, :] = ds_height[i, ds_mask_e3].flatten()
# mean_ds_height = mean_ds_height/hm_m_model


################################ real froude number
# froude_number_2010080320_0907 = xr.open_dataset(
#     'scratch/simulation/20100801_09_3d/02_lm_post_processed/froude_number_2010080320_0907.nc')
# (froude_number_2010080320_0907.time.values == time).all()
# (froude_number_2010080320_0907.lon.values == lon).all()
# (froude_number_2010080320_0907.lat.values == lat).all()

# fr = froude_number_2010080320_0907.fr.values
# mean_fr = np.ones((len(time), np.sum(ds_mask_e2)))
# for i in np.arange(0, len(time)):
#     mean_fr[i, :] = fr[i, ds_mask_e2].flatten()

# h_fr = (1 - mean_fr)*hm_m_model


################################ wind velocity
velocity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')

u_earth = velocity_3d_20100801_09z.u_earth[68:200, 1, :, :].values
v_earth = velocity_3d_20100801_09z.v_earth[68:200, 1, :, :].values

# u_earth_e1 = np.ones((len(time), np.sum(mask_e1)))
# v_earth_e1 = np.ones((len(time), np.sum(mask_e1)))
u_earth_e2 = np.ones((len(time), np.sum(mask_e2)))
v_earth_e2 = np.ones((len(time), np.sum(mask_e2)))
u_earth_e3 = np.ones((len(time), np.sum(mask_e3)))
v_earth_e3 = np.ones((len(time), np.sum(mask_e3)))
for i in np.arange(0, len(time)):
    # u_earth_e1[i, :] = u_earth[i, mask_e1].flatten()
    # v_earth_e1[i, :] = v_earth[i, mask_e1].flatten()
    u_earth_e2[i, :] = u_earth[i, mask_e2].flatten()
    v_earth_e2[i, :] = v_earth[i, mask_e2].flatten()
    u_earth_e3[i, :] = u_earth[i, mask_e3].flatten()
    v_earth_e3[i, :] = v_earth[i, mask_e3].flatten()


################################ dimensionless mountain height
h_dim = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/h_dim_20100801_09.nc'
    )
h_dim_values = h_dim.h_dim[68:200, :, :].values
h_dim_e3 = np.ones((len(time), np.sum(mask_e3)))
for i in np.arange(0, len(time)):
    h_dim_e3[i, :] = h_dim_values[i, mask_e3].flatten()


################################ inversion base
inversion_height_20100801_09 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc'
)

inversion_height = inversion_height_20100801_09.inversion_height[
    68:200, :, :].values
inversion_height_e3 = np.ones((len(time), np.sum(mask_e3)))
for i in np.arange(0, len(time)):
    inversion_height_e3[i, :] = inversion_height[i, mask_e3].flatten()


################################ number of vortices

identified_count = np.zeros(len(time))
for i in range(len(time)):
    identified_count[i] = len(correctly_reidentified[str(time[i])[0:13]])

################################ inversion base in ERA 5
# -16.937720915263164; 33.01360906782903
# slev_bLheight.longitude[26].values
# slev_bLheight.latitude[7].values
# slev_bLheight = xr.open_dataset(
#     'scratch/obs/era5/single_level_BLheight_20100801_09.nc'
#     )
# slev_bLheight.blh[:, 7, 26].values

################################ PBL height in model
# filelist = np.array(sorted( glob.glob(
#     '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd2010080[1-9]*0000.nc'
#     )))
# ncfiles = xr.open_mfdataset(
#     filelist, concat_dim="time",
#     data_vars='minimal', coords='minimal', compat='override'
# ).metpy.parse_cf()

# hpbl = ncfiles.HPBL[:, 80:920, 80:920].values
# hpbl_e2 = np.ones((len(time), np.sum(mask_e2)))
# for i in np.arange(0, len(time)):
#     hpbl_e2[i, :] = hpbl[i, mask_e2].flatten()
# stats.describe(hpbl_e2.flatten())

################################ hinv from Funchal and nearest grid
funchal_inversion_height = pd.read_pickle(
    'scratch/inversion_height/radiosonde/funchal_inversion_height.pkl')
ifunchal1 = np.where(
    (funchal_inversion_height.index >= min(time)) &
    (funchal_inversion_height.index <= max(time)))[0]
# funchal_inversion_height.index[ifunchal1]

# min(time), max(time)
filelist = np.array(sorted(glob.glob(
    'scratch/inversion_height/madeira3d/inversion_height_madeira3d_20*.nc')))
inversion_height_madeira3d = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override')
funchal_loc = np.array([32.6333, -16.9000])
nearestgrid_indices = find_nearest_grid(
    funchal_loc[0], funchal_loc[1],
    inversion_height_madeira3d.lat.values,
    inversion_height_madeira3d.lon.values)
inversion_height_nearestgrid = pd.DataFrame(
    data=inversion_height_madeira3d.inversion_height[
        :, nearestgrid_indices[0], nearestgrid_indices[1]
    ].values,
    index = inversion_height_madeira3d.time.values,
    columns=['inversion_height'],)
ifunchal2 = np.where(
    (inversion_height_nearestgrid.index >= min(time)) &
    (inversion_height_nearestgrid.index <= max(time)))[0]
inversion_height_nearestgrid.index[ifunchal2]


ymax = 4000
y_e3 = 3600  # [3200-4000]
y_e2 = 3000  # [2600-3200]
y_hdim = 1750  # [1600-2600]
windstd_int = 150
count_int = 150
hdim_int = 50
mpl.rc('font', family='Times New Roman', size=12)
fig, ax = plt.subplots(1, 1, figsize=np.array([14, 12]) / 2.54)

################################ plot BLH from ERA5 and model
# ax.plot(
#     time, slev_bLheight.blh[68:200, 7, 26].values, linewidth=1,
#     color='blue', linestyle='--',)
# ax.plot(
#     time, np.mean(hpbl_e2, axis = 1), linewidth=1,
#     color='blue', linestyle=':',)

################################ plot hinv from Funchal and nearest grid
ax.plot(
    funchal_inversion_height.index[ifunchal1],
    funchal_inversion_height.values[ifunchal1],
    linewidth=1, color='blue', linestyle='--', marker = 'o', markersize=3.5,)
ax.plot(
    inversion_height_nearestgrid.index[ifunchal2],
    inversion_height_nearestgrid.values[ifunchal2],
    linewidth=1, color='blue', linestyle=':')

################################ plot number of vortices
ax.plot(
    time, y_hdim + identified_count * count_int, linewidth=1, color='g'
    )

################################ plot dimensionless mountain height

ax.plot(
    time, y_hdim + np.mean(h_dim_e3, axis=1) * hdim_int, linewidth=1,
    color='r'
    )
# ax.fill_between(
#     time,
#     y_hdim + (np.mean(h_dim_e3, axis=1) - 2 *
#               np.std(h_dim_e3, axis=1)) * hdim_int,
#     y_hdim + (np.mean(h_dim_e3, axis=1) + 2 *
#               np.std(h_dim_e3, axis=1)) * hdim_int,
#     color='gray', alpha=0.2, edgecolor=None,
#     )
plt.axhline(y=y_hdim + count_int * 1, color='black', linestyle='--', lw=0.5)
plt.axvline(x=time[141-68], color='black', linestyle='--', lw=0.5)

################################ plot dividing streamline
plt_ds, = ax.plot(
    time, np.mean(mean_ds_height, axis = 1), linewidth=1, color='black'
    )
# plt_ds_std = ax.fill_between(
#     time,
#     np.mean(mean_ds_height, axis=1) - 2 * np.std(mean_ds_height, axis=1),
#     np.mean(mean_ds_height, axis=1) + 2 * np.std(mean_ds_height, axis=1),
#     color='gray', alpha=0.2, edgecolor=None,
#     )
plt.axhline(y=hm_m_model * 0.6, color='black', linestyle='--', lw=0.5)
# plt.axhline(y=y_e3, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
# plt.axhline(y=y_e2, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')

################################ plot froude number
# plt_fr, = ax.plot(
#     time, np.mean(h_fr, axis=1), linewidth=1, color='black',
#     linestyle='dashed',
# )

################################ plot inversion base layer
plt_inv, = ax.plot(
    time, np.mean(inversion_height_e3, axis=1), linewidth=1, color='blue')
# plt_inv_std = ax.fill_between(
#     time,
#     np.mean(inversion_height_e3, axis=1) - 2 *
#     np.std(inversion_height_e3, axis=1),
#     np.mean(inversion_height_e3, axis=1) + 2 *
#     np.std(inversion_height_e3, axis=1),
#     color='gray', alpha=0.2, edgecolor=None,)

################################ plot wind in e3
plt_quiver = ax.quiver(time, y_e3,
          np.mean(u_earth_e3, axis=1), np.mean(v_earth_e3, axis=1),
          rasterized=True, units='height', scale=150, width=0.002,
          headwidth=3, headlength=5, alpha=0.75)
ax.quiverkey(plt_quiver, X=0.1, Y=0.96, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E',
             labelsep = 0.1)
plt.axhline(y=y_e3, color='gray', linestyle='--', lw=0.1)

# ax.fill_between(
#     time,
#     y_e3,
#     y_e3 + windstd_int * np.sqrt(np.var(u_earth_e3, axis=1) +
#                                  np.var(v_earth_e3, axis=1)),
#     color='gray', alpha=0.2, edgecolor=None,)

################################ plot wind in e2
ax.quiver(time, y_e2,
          np.mean(u_earth_e2, axis=1), np.mean(v_earth_e2, axis=1),
          rasterized=True, units='height', scale=150, width=0.002,
          headwidth=3, headlength=5, alpha=0.75,)
plt.axhline(y=y_e2, color='gray', linestyle='--', lw=0.1)

# ax.fill_between(
#     time,
#     y_e2,
#     y_e2 + windstd_int * np.sqrt(np.var(u_earth_e2, axis=1) +
#                                  np.var(v_earth_e2, axis=1)),
#     color='gray', alpha=0.2, edgecolor=None,)

################################ set yticks
ax.set_ylim(0, ymax)
ax.set_yticks(
    np.concatenate((
        np.arange(0, 1501, 300),
        np.arange(y_hdim, y_hdim + count_int * 5 + 1, count_int),
        # np.arange(y_e2, y_e2 + windstd_int * 2 + 1, windstd_int),
        # np.arange(y_e3, y_e3 + windstd_int * 1 + 1, windstd_int),
        )))
ax.set_yticklabels(
    ['0', '0.3', '0.6', '0.9', '1.2', '1.5',
     '0', '1', '2', '3', '4', '5',
    #  '0', '1', '2',
    #  '0', '1',
     ])

################################ set xticks
time_tick = pd.date_range('2010-08-04', '2010-08-09', freq="12h")
ax.set_xticks(time_tick)
ax.set_xticklabels(
    ['04-00', '04-12', '05-00', '05-12', '06-00', '06-12',
     '07-00', '07-12', '08-00', '08-12', '09-00', ],
    rotation = 45)

################################ create a second axes for Fr
ax2 = ax.twinx()
ax2.set_ylim(0, ymax)
ax2.set_yticks(
    np.concatenate((
        # np.arange(0, 1.1, 0.2) * hm_m_model,
        np.arange(y_hdim, y_hdim + hdim_int * 9 + 1, hdim_int * 3),
        )))
ax2.set_yticklabels([
    # '1.0', '0.8', '0.6', '0.4', '0.2', '0.0',
     '0', '3', '6', '9',
     ])

################################ add labels
# ax2.text(time[-1] + np.timedelta64(18, 'h'),
#          650, '$Fr_{ds} \;\; [-]$', rotation=90)
ax2.text(time[-1] + np.timedelta64(18, 'h'), 1750, '$h_{dim} \;\; [-]$',
         rotation=90, color='r')
ax.text(time[0] - np.timedelta64(23, 'h'), 800, "$h_{ds} \;\; [km]$", rotation=90)
ax.text(time[0] - np.timedelta64(23, 'h'), 400, "$h_{inv},$",
        rotation=90, color='blue')
ax.text(time[0] - np.timedelta64(23, 'h'), 1800, "Vortices [#]",
        rotation=90, color='g')
ax.text(time[120], y_e2 + 1 * windstd_int,
        r"$e_2$",
        # r"$e_2 : \sqrt{\sigma_u^2 + \sigma_v^2} \; [m \; s^{-1}]$",
        rotation=0)
ax.text(time[120], y_e3 + 1 * windstd_int,
        r"$e_3$",
        # r"$e_3 : \sqrt{\sigma_u^2 + \sigma_v^2} \; [m \; s^{-1}]$",
        rotation=0)
ax.set_xlabel('Atmospheric flow conditions during 03-09 August 2010')
ax.grid(True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.16, top=0.995)

# fig.savefig('figures/00_test/trial.png', dpi=600)
fig.savefig(
    'figures/06_case_study/6.2.0 flow conditions during 201008 0320_0907.png',
    dpi=600)
plt.close('all')


'''
########
wind_earth_1h_100m_average = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_average_2010.nc')


# time = pd.date_range('2009-12-31', '2010-11-30', freq="1M")
time = [pd.to_datetime('2010-' + i + '-01',
                       infer_datetime_format=True) for i in months]



######## plot masked area

extentm_up = [
    np.min(lon), -16.25,
    32.4, np.max(lat)]
ticklabelm_up = ticks_labels(-17.3, -16.3, 32.4, 33.2, 0.2, 0.2)
fig, ax = framework_plot(
    "self_defined", figsize=np.array([8.8, 7.5]) / 2.54, lw=0.25, labelsize=8,
    extent=extentm_up, ticklabel=ticklabelm_up
)
demlevel = np.arange(0, 0.301, 0.001)
ticks = np.arange(0, 0.301, 0.05)
plt_dem = ax.pcolormesh(
    lon, lat,
    froude_n,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('RdBu', len(demlevel)), rasterized=True,
    transform=transform,
)
cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.12, fraction=0.12,
    shrink=0.8, aspect=25, ticks=ticks, extend='neither')
cbar.ax.set_xlabel("Froude number at 20100803 20:00 UTC [-]")

ellipse2 = Ellipse(
    [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
        center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse2)

# ds_masked_e2 = np.ma.ones(lon.shape)
# ds_masked_e2[~ds_mask_e2] = np.ma.masked
# model_lb2 = ax.contourf(
#     lon, lat, ds_masked_e2,
#     transform=transform, colors='gainsboro', alpha=1, zorder=2)

scale_bar(ax, bars=2, length=20, location=(0.05, 0.025),
          barheight=1, linewidth=0.2, col='black')

ax.set_extent(extentm_up, crs=transform)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.1, top=0.98)
fig.savefig('figures/00_test/trial.png', dpi=600)

'''

# endregion
# =============================================================================




