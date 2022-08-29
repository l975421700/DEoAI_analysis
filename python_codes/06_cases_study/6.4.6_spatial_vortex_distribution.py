

# =============================================================================
# region import packages


# basic library
from matplotlib.path import Path
import pywt.data
import pywt
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc
from scipy import stats
import tables as tb

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
# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
    framework_plot1,
)


# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
from scipy import ndimage
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py

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
    rvor_level,
    rvor_ticks,
    rvor_cmp,
    center_madeira,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    sig_coeffs,
    vortex_identification,
    vortex_identification1,
)


# endregion
# =============================================================================


smaller_domain = True
# =============================================================================
# region aggregate vorticity grids

#### potential region around Madeira: based on indices
nc1h_second_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20051101000000c.nc')
lon_a = nc1h_second_c.lon.values
lat_a = nc1h_second_c.lat.values
lon = lon_a[80:920, 80:920]
lat = lat_a[80:920, 80:920]

from matplotlib.path import Path
## first set of boundary
polygon=[
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),]
poly_path=Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(840, 840)
mask_data = np.zeros(lon_a.shape)
mask_data[80:920, 80:920][mask] = 1
# mask_data1 = np.ma.array(np.ones(mask.shape))
# mask_data1.mask = (mask == False)
## second set of boundary
polygon1 = [
    (390, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0), ]
poly_path1 = Path(polygon1)
mask1 = poly_path1.contains_points(coors).reshape(840, 840)
mask_data1 = np.zeros(lon_a.shape)
mask_data1[80:920, 80:920][mask1] = 1

#### old simulation constant file
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

# grid_vortex_count = np.zeros(lon.shape, dtype = np.int64)

# for i in range(10):
#     # i = 4
#     if smaller_domain:
#         inputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_past_sd/identified_transformed_rvor_20" + \
#             years[i] + ".h5"
#     else:
#         inputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_past/identified_transformed_rvor_20" + years[i] + ".h5"
#     identified_rvor = tb.open_file(inputfile, mode="r")
#     experiment = identified_rvor.root.exp1
#     annual_is_vortex = experiment.vortex_info.cols.is_vortex[:].copy()
#     sum_is_vortex = annual_is_vortex.sum(axis=(1,2))
#     # stats.describe(sum_is_vortex)
#     # filter hourly isolated vortices
#     previous_sum = np.concatenate((np.array([0]), sum_is_vortex[:-1]))
#     next_sum = np.concatenate((sum_is_vortex[1:], np.array([0]), ))
#     isolated_hour = np.vstack(((sum_is_vortex > 0), (previous_sum == 0),
#                                (next_sum == 0))).all(axis=0)
#     filtered_annual_is_vortex = annual_is_vortex.copy()
#     filtered_annual_is_vortex[isolated_hour, :, :] = 0
#     grid_vortex_count += filtered_annual_is_vortex.sum(axis = 0)
#     # np.sum(grid_vortex_count == filtered_annual_is_vortex.sum(axis=0) )
#     identified_rvor.close()
#     print(i)

# if smaller_domain:
#     np.save(
#         'scratch/rvorticity/rvor_identify/re_identify/grid_vortex_count2006_2015_sd.npy',
#         grid_vortex_count,)
# else:
#     np.save(
#         'scratch/rvorticity/rvor_identify/re_identify/grid_vortex_count2006_2015.npy',
#         grid_vortex_count,)

if smaller_domain:
    grid_vortex_count = np.load(
        'scratch/rvorticity/rvor_identify/re_identify/grid_vortex_count2006_2015_sd.npy')
else:
    grid_vortex_count = np.load(
        'scratch/rvorticity/rvor_identify/re_identify/grid_vortex_count2006_2015.npy')


################ plot count
level = np.arange(0, 800.1, 2)
ticks = np.arange(0, 800.1, 200)

fig, ax = framework_plot1("1km_lb",)
plt_count = ax.pcolormesh(
    lon, lat, grid_vortex_count, cmap=cm.get_cmap('Blues', len(level)),
    norm=BoundaryNorm(level, ncolors=len(level), clip=False),
    transform=transform, rasterized=True, zorder=-2,)
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
# ax.contourf(
#     lon, lat, mask_data1,
#     transform=transform, colors='gainsboro', alpha=0.5, zorder=3)
ax.contour(
    lon_a, lat_a, mask_data, colors='m', levels=np.array([0.5]),
    linewidths=1, linestyles='solid')
ax.contour(
    lon_a, lat_a, mask_data1, colors='r', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
cbar = fig.colorbar(
    plt_count, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Decadal vortex count [-]")

if smaller_domain:
    fig.savefig(
        'figures/09_decades_vortex/09_07_smaller_domain/9_7.0.0_vortex_spatial_distribution_sd.png')
else:
    fig.savefig(
        'figures/09_decades_vortex/09_07_smaller_domain/9_7.0.1_vortex_spatial_distribution.png')


################ plot log(count)
stats.describe(grid_vortex_count.flatten())
grid_vortex_count_ln = np.log(grid_vortex_count)
stats.describe(grid_vortex_count_ln.flatten(), nan_policy = 'omit')
grid_vortex_count_ln[np.isfinite(grid_vortex_count_ln) == False] = np.nan

level = np.arange(0, 7.001, 0.02)
ticks = np.arange(0, 7.001, 1)

fig, ax = framework_plot1("1km_lb", dpi = 600)
plt_count = ax.pcolormesh(
    lon, lat, grid_vortex_count_ln, cmap=cm.get_cmap('Blues', len(level)),
    norm=BoundaryNorm(level, ncolors=len(level), clip=False),
    transform=transform, rasterized=True, zorder=-2,)
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
ax.contour(
    lon_a, lat_a, mask_data, colors='black', levels=np.array([0.5]),
    linewidths=1, linestyles='dashed')
ax.contour(
    lon_a, lat_a, mask_data1, colors='r', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
cbar = fig.colorbar(
    plt_count, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='min',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Hourly vortex count [-] from 2006 to 2015")
cbar.ax.set_xticklabels(['$e^0$', '$e^1$', '$e^2$', '$e^3$',
                         '$e^4$', '$e^5$', '$e^6$', '$e^7$', ])

if smaller_domain:
    fig.savefig(
        'figures/09_decades_vortex/09_07_smaller_domain/9_7.0.2_log_vortex_spatial_distribution_sd.png')
else:
    fig.savefig(
        'figures/09_decades_vortex/09_07_smaller_domain/9_7.0.3_log_vortex_spatial_distribution.png')




################ plot frequency
# stats.describe(grid_vortex_count.flatten())
# grid_vortex_frequency = grid_vortex_count/87648
# stats.describe(grid_vortex_frequency.flatten())

'Hourly vortex frequency [%]'

'''
# check
grid_vortex_count_sd = np.load(
    'scratch/rvorticity/rvor_identify/re_identify/grid_vortex_count2006_2015_sd.npy')
grid_vortex_count = np.load(
    'scratch/rvorticity/rvor_identify/re_identify/grid_vortex_count2006_2015.npy')
np.sum(grid_vortex_count_sd == grid_vortex_count)

level = np.arange(0, 10.1, 0.1)
ticks = np.arange(0, 10.1, 2)

fig, ax = framework_plot1("1km_lb",)
plt_count = ax.pcolormesh(
    lon, lat, grid_vortex_count - grid_vortex_count_sd,
    cmap=cm.get_cmap('Blues', len(level)),
    norm=BoundaryNorm(level, ncolors=len(level), clip=False),
    transform=transform, rasterized=True, zorder=-2,)
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
ax.contour(
    lon_a, lat_a, mask_data, colors='m', levels=np.array([0.5]),
    linewidths=1, linestyles='solid')
ax.contour(
    lon_a, lat_a, mask_data1, colors='r', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
cbar = fig.colorbar(
    plt_count, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Decadal vortex count difference [-]")
fig.savefig('figures/00_test/trial.png')





stats.describe(grid_vortex_count.flatten())

(previous_sum[1:] == sum_is_vortex[:-1]).all()
(next_sum[:-1] == sum_is_vortex[1:]).all()
np.sum(isolated_hour)
np.sum(filtered_annual_is_vortex), np.sum(annual_is_vortex)
filtered_sum_is_vortex = filtered_annual_is_vortex.sum(axis=(1, 2))
np.sum(filtered_sum_is_vortex != sum_is_vortex)

#### wind data
wind_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20*.nc')))
wind_earth_1h_100m = xr.open_mfdataset(
    wind_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = wind_earth_1h_100m.time.values
lon = wind_earth_1h_100m.lon.values
lat = wind_earth_1h_100m.lat.values

'''
# endregion
# =============================================================================






