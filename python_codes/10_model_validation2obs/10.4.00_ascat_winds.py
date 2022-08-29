

# =============================================================================
# region import packages


# basic library
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
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py
from scipy.ndimage import median_filter

from DEoAI_analysis.module.vortex_namelist import (
    correctly_identified
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


# =============================================================================
# region plot global ascat winds

daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1

######## original simulation to calculate surface theta
filelist = np.array(sorted(glob.glob('data_source/eumetsat/1505036-1of1/*.nc')))
ncfile = xr.open_dataset(filelist[0])
lon = ncfile.lon.values
lat = ncfile.lat.values
wind_speed = ncfile.wind_speed.values

# ddd = lon[1500, 1:] - lon[1500, :-1]
# ddd[40]
# lon[1500, 41] - lon[1500, 40]
# ccc = lon[0, 1:] - lon[0, :-1]

windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

fig, ax = framework_plot(
    "global", figsize=np.array([17.6, 10.24]) / 2.54, lw=0.1, labelsize=10)
plt_wind1 = ax.pcolormesh(
    lon[:, 0:41], lat[:, 0:41], wind_speed[:, 0:41][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
plt_wind2 = ax.pcolormesh(
    lon[:, 41:], lat[:, 41:], wind_speed[:, 41:][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
ax.contour(
    lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
    linewidths=0.25, linestyles='solid')

cbar = fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06,
    shrink=0.5, aspect=30, ticks=ticks, extend='max',)
# cbar.ax.set_xlabel(
#     'Wind velocity [$m \; s^{-1}$]' + ' from ' + str(ncfile.start_date) + \
#         ' ' + str(ncfile.start_time) + ' to ' + \
#             str(ncfile.stop_date) + ' ' + str(ncfile.stop_time))
ax.text(
    0, -136,
    'Wind velocity [$m \; s^{-1}$]' + ' from ' + str(ncfile.start_date) +
    ' ' + str(ncfile.start_time) + ' to ' +
    str(ncfile.stop_date) + ' ' + str(ncfile.stop_time),
    horizontalalignment='center')
fig.subplots_adjust(left=0.05, right=0.975, bottom=0.10, top=0.98)
fig.savefig('figures/10_validation2obs/10_03_winds/10_03.0.0 Global winds in ascat at 2010080322.png', dpi=600)


'''


ncfile.wind_dir.values
'''
# endregion
# =============================================================================


# =============================================================================
# region animation global ascat winds

######## input analysis region
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1

windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

######## original simulation to calculate surface theta
# filelist = np.array(sorted(glob.glob('data_source/eumetsat/1505036-1of1/*.nc')))
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.0.1 Global winds around madeira in ascat at 20100803_09.mp4'
# filelist = np.array(sorted(glob.glob(
#     'data_source/eumetsat/1505036-1of1_backup/*.nc')))
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.0.2 Global winds around madeira_all in ascat at 20100803_09.mp4'
# filelist = np.array(sorted(glob.glob(
#     'data_source/eumetsat/1507216-1of1/*.nc')))
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.0.3 Global winds in ascat at 20100803_09.mp4'
filelist = np.array(sorted(glob.glob(
    'data_source/eumetsat/1508158-1of1/*.nc')))
outputfile = 'figures/10_validation2obs/10_03_winds/10_03.0.5 Global winds in ascat at 20100820_31.mp4'

######## animate
fig, ax = framework_plot(
    "global", figsize=np.array([17.6, 10.24]) / 2.54,
    lw=0.1, labelsize=10)

ims =[]

for i in range(len(filelist)):  # range(2):  #
    # i = 0
    ncfile = xr.open_dataset(filelist[i])
    
    lon = ncfile.lon.values
    lat = ncfile.lat.values
    wind_speed = ncfile.wind_speed.values
    
    plt_wind1 = ax.pcolormesh(
        lon[:, 0:41], lat[:, 0:41], wind_speed[:, 0:41][:-1, :-1],
        cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        animated=True)
    plt_wind2 = ax.pcolormesh(
        lon[:, 41:], lat[:, 41:], wind_speed[:, 41:][:-1, :-1],
        cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        animated=True)
    plt_ana_region = ax.contour(
        lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
        linewidths=0.25, linestyles='solid')
    plt_text = ax.text(
        0, -136,
        'Wind velocity [$m \; s^{-1}$]' + ' from ' + str(ncfile.start_date) +
        ' ' + str(ncfile.start_time) + ' to ' +
        str(ncfile.stop_date) + ' ' + str(ncfile.stop_time),
        horizontalalignment='center')
    ims.append([plt_wind1, plt_wind2, plt_text] + plt_ana_region.collections)
    print(str(i) + '/' + str(len(filelist) - 1))

fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06,
    shrink=0.5, aspect=30, ticks=ticks, extend='max',)
fig.subplots_adjust(left=0.05, right=0.975, bottom=0.10, top=0.98)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'), dpi=600)



# endregion
# =============================================================================


# =============================================================================
# region plot regional ascat winds around madeira

#### create a boundary
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

#### mask outside
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

####
filelist = np.array(sorted(glob.glob('data_source/eumetsat/1505036-1of1/*.nc')))

# i = 0
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.1.0 Regional winds in ascat at 2010080322.png'
# i = 3
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.1.1 Regional winds in ascat at 2010080509.png'
i = 4
outputfile = 'figures/10_validation2obs/10_03_winds/10_03.1.3 Regional winds in ascat at 2010080521.png'

ncfile = xr.open_dataset(filelist[i])
lon = ncfile.lon.values
lat = ncfile.lat.values
wind_speed = ncfile.wind_speed.values
time = ncfile.time.values

#### madaira time
lon_m = lon.copy()
lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
coors_m = np.hstack((lon_m.reshape(-1, 1), lat.reshape(-1, 1)))
mask_m = poly_path.contains_points(coors_m).reshape(
    lon.shape[0], lon.shape[1])
# time[mask_m]

windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

fig, ax = framework_plot1("1km_lb", figsize=np.array([8.8, 9.3]) / 2.54,)
plt_wind1 = ax.pcolormesh(
    lon[:, 0:41], lat[:, 0:41], wind_speed[:, 0:41][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
plt_wind2 = ax.pcolormesh(
    lon[:, 41:], lat[:, 41:], wind_speed[:, 41:][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
ax.contour(
    lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
    linewidths=0.25, linestyles='solid')
ax.contourf(mask_lon2, mask_lat2, masked,
            colors='white', levels=np.array([0.5, 1.5]))
cbar = fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1.5), panchor=(0.5, 0))
# cbar.ax.set_xlabel("Wind velocity [$m\;s^{-1}$]", fontsize=8)
cbar.ax.set_xlabel(
    '10-meter wind velocity [$m\;s^{-1}$] by ASCAT' + '\nfrom' +
    ' ' + str(np.min(time[mask_m]))[0:19] + ' to ' +
    str(np.max(time[mask_m]))[0:19], fontsize=10)

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)
fig.savefig(outputfile, dpi=600)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate regional ascat winds around madeira

#### create a boundary
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

#### mask outside
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

#### input data and plot
windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

# filelist = np.array(sorted(glob.glob(
#     'data_source/eumetsat/1505036-1of1_backup/*.nc')))
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.1.2 Regional winds around madeira_all in ascat at 20100803_09.mp4'
filelist = np.array(sorted(glob.glob(
    'data_source/eumetsat/1507216-1of1/*.nc')))
outputfile = 'figures/10_validation2obs/10_03_winds/10_03.1.4 Regional winds in ascat at 20100803_09.mp4'

fig, ax = framework_plot1("1km_lb",)
ims =[]

for i in range(len(filelist)):  # range(2):  #
    # i = 0
    ncfile = xr.open_dataset(filelist[i])
    lon = ncfile.lon.values
    lat = ncfile.lat.values
    wind_speed = ncfile.wind_speed.values
    
    plt_wind1 = ax.pcolormesh(
        lon[:, 0:41], lat[:, 0:41], wind_speed[:, 0:41][:-1, :-1],
        cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        animated=True)
    plt_wind2 = ax.pcolormesh(
        lon[:, 41:], lat[:, 41:], wind_speed[:, 41:][:-1, :-1],
        cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        animated=True)
    
    plt_ana_region = ax.contour(
        lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
        linewidths=0.25, linestyles='solid')
    plt_mask = ax.contourf(
        mask_lon2, mask_lat2, masked,
        colors='white', levels=np.array([0.5, 1.5]))
    plt_text = ax.text(
        -18, 21,
        'Wind velocity [$m \; s^{-1}$]' + ' from ' + str(ncfile.start_date) +
        ' ' + str(ncfile.start_time) + ' to ' +
        str(ncfile.stop_date) + ' ' + str(ncfile.stop_time),
        horizontalalignment='center', fontsize=8)
    ims.append(
        [plt_wind1, plt_wind2, plt_text] + \
        plt_ana_region.collections + plt_mask.collections)
    print(str(i) + '/' + str(len(filelist) - 1))

fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'), dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region calculate and plot relative vorticity

#### create a boundary
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

#### mask outside
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

#### input ascat winds
filelist = np.array(
    sorted(glob.glob('data_source/eumetsat/1505036-1of1/*.nc')))

# i = 0
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.2.0 Relative vorticity in ascat at 2010080322.png'
# i = 3
# outputfile = 'figures/10_validation2obs/10_03_winds/10_03.2.1 Relative vorticity in ascat at 2010080509.png'
i = 4
outputfile = 'figures/10_validation2obs/10_03_winds/10_03.2.2 Relative vorticity in ascat at 2010080521.png'

ncfile = xr.open_dataset(filelist[i])
lon = ncfile.lon.values
lat = ncfile.lat.values
wind_speed = ncfile.wind_speed.values
wind_dir = ncfile.wind_dir.values
time = ncfile.time.values

lon_m = lon.copy()
lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
coors_m = np.hstack((lon_m.reshape(-1, 1), lat.reshape(-1, 1)))
mask_m = poly_path.contains_points(coors_m).reshape(
    lon.shape[0], lon.shape[1])
# time[mask_m]

#### calculate relative vorticity

lon_s1 = lon[:, 0:41]
lat_s1 = lat[:, 0:41]
wind_speed_s1 = wind_speed[:, 0:41]
wind_dir_s1 = wind_dir[:, 0:41]
wind_u1 = wind_speed_s1 * np.sin(np.deg2rad(wind_dir_s1))
wind_v1 = wind_speed_s1 * np.cos(np.deg2rad(wind_dir_s1))
dx1, dy1 = mpcalc.lat_lon_grid_deltas(lon_s1, lat_s1)
rvor1 = mpcalc.vorticity(
    wind_u1 * units('m/s'), wind_v1 * units('m/s'),
    dx1, dy1, dim_order='yx') * 10**4

lon_s2 = lon[:, 41:]
lat_s2 = lat[:, 41:]
wind_speed_s2 = wind_speed[:, 41:]
wind_dir_s2 = wind_dir[:, 41:]
wind_u2 = wind_speed_s2 * np.sin(np.deg2rad(wind_dir_s2))
wind_v2 = wind_speed_s2 * np.cos(np.deg2rad(wind_dir_s2))
dx2, dy2 = mpcalc.lat_lon_grid_deltas(lon_s2, lat_s2)
rvor2 = mpcalc.vorticity(
    wind_u2 * units('m/s'), wind_v2 * units('m/s'),
    dx2, dy2, dim_order='yx') * 10**4

#### plot relative vorticity
fig, ax = framework_plot1("1km_lb", figsize=np.array([8.8, 9.3]) / 2.54,)

rvor_level = np.arange(-6, 6.01, 0.1)
rvor_ticks = np.arange(-6, 6.1, 2)
rvor_top = cm.get_cmap('Blues_r', int(np.floor(len(rvor_level) / 2)))
rvor_bottom = cm.get_cmap('Reds', int(np.floor(len(rvor_level) / 2)))
rvor_colors = np.vstack(
    (rvor_top(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2)))),
     [1, 1, 1, 1],
     rvor_bottom(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2))))))
rvor_cmp = ListedColormap(rvor_colors, name='RedsBlues_r')

plt_rvor1 = ax.pcolormesh(
    lon_s1, lat_s1, rvor1, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),
    zorder=-2,)
plt_rvor2 = ax.pcolormesh(
    lon_s2, lat_s2, rvor2, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),
    zorder=-2,)

cbar = fig.colorbar(
    plt_rvor2, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=rvor_ticks, extend='both',
    anchor=(0.5, 1.5), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    '10-meter relative vorticity [$10^{-4}\;s^{-1}$] by ASCAT' + '\nfrom' + \
    ' ' + str(np.min(time[mask_m]))[0:19] + ' to ' + \
    str(np.max(time[mask_m]))[0:19], fontsize=10)

# ax.contour(
#     lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
#     linewidths=0.25, linestyles='solid')
ax.contourf(mask_lon2, mask_lat2, masked,
            colors='white', levels=np.array([0.5, 1.5]))
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)
fig.savefig(outputfile, dpi=600)



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region calculate and animate relative vorticity

#### create a boundary
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

#### mask outside
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

#### input ascat winds
filelist = np.array(sorted(glob.glob(
    'data_source/eumetsat/1505036-1of1_backup/*.nc')))
outputfile = 'figures/10_validation2obs/10_03_winds/10_03.2.3 Relative vorticity in ascat at 20100803_09.mp4'

rvor_level = np.arange(-6, 6.01, 0.1)
rvor_ticks = np.arange(-6, 6.1, 2)
rvor_top = cm.get_cmap('Blues_r', int(np.floor(len(rvor_level) / 2)))
rvor_bottom = cm.get_cmap('Reds', int(np.floor(len(rvor_level) / 2)))
rvor_colors = np.vstack(
    (rvor_top(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2)))),
     [1, 1, 1, 1],
     rvor_bottom(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2))))))
rvor_cmp = ListedColormap(rvor_colors, name='RedsBlues_r')

fig, ax = framework_plot1("1km_lb", figsize=np.array([8.8, 9.3]) / 2.54,)
ims =[]

for i in range(len(filelist)):  # range(2): #
    # i=0
    #### input data
    ncfile = xr.open_dataset(filelist[i])
    lon = ncfile.lon.values
    lat = ncfile.lat.values
    wind_speed = ncfile.wind_speed.values
    wind_dir = ncfile.wind_dir.values
    time = ncfile.time.values
    
    lon_m = lon.copy()
    lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
    coors_m = np.hstack((lon_m.reshape(-1, 1), lat.reshape(-1, 1)))
    mask_m = poly_path.contains_points(coors_m).reshape(
        lon.shape[0], lon.shape[1])
    
    #### calculate relative vorticity
    lon_s1 = lon[:, 0:41]
    lat_s1 = lat[:, 0:41]
    wind_speed_s1 = wind_speed[:, 0:41]
    wind_dir_s1 = wind_dir[:, 0:41]
    wind_u1 = wind_speed_s1 * np.sin(np.deg2rad(wind_dir_s1))
    wind_v1 = wind_speed_s1 * np.cos(np.deg2rad(wind_dir_s1))
    dx1, dy1 = mpcalc.lat_lon_grid_deltas(lon_s1, lat_s1)
    rvor1 = mpcalc.vorticity(
        wind_u1 * units('m/s'), wind_v1 * units('m/s'),
        dx1, dy1, dim_order='yx') * 10**4
    
    lon_s2 = lon[:, 41:]
    lat_s2 = lat[:, 41:]
    wind_speed_s2 = wind_speed[:, 41:]
    wind_dir_s2 = wind_dir[:, 41:]
    wind_u2 = wind_speed_s2 * np.sin(np.deg2rad(wind_dir_s2))
    wind_v2 = wind_speed_s2 * np.cos(np.deg2rad(wind_dir_s2))
    dx2, dy2 = mpcalc.lat_lon_grid_deltas(lon_s2, lat_s2)
    rvor2 = mpcalc.vorticity(
        wind_u2 * units('m/s'), wind_v2 * units('m/s'),
        dx2, dy2, dim_order='yx') * 10**4
    
    plt_rvor1 = ax.pcolormesh(
        lon_s1, lat_s1, rvor1, cmap=rvor_cmp, rasterized=True,
        transform=transform,
        norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),
        zorder=-2,)
    plt_rvor2 = ax.pcolormesh(
        lon_s2, lat_s2, rvor2, cmap=rvor_cmp, rasterized=True,
        transform=transform,
        norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),
        zorder=-2,)
    
    if (np.sum(mask_m) != 0):
        plt_text = ax.text(
            -18, 20.5,
            'Relative vorticity [$10^{-4}\;s^{-1}$]' + '\nfrom' +
            ' ' + str(np.min(time[mask_m]))[0:19] + ' to ' +
            str(np.max(time[mask_m]))[0:19],
            horizontalalignment='center', fontsize=10)
    else:
        plt_text = ax.text(
            -18, 21,
            'Relative vorticity [$10^{-4}\;s^{-1}$]',
            horizontalalignment='center', fontsize=10)
    
    ims.append([plt_rvor1, plt_rvor2, plt_text])
    
    print(str(i) + '/' + str(len(filelist) - 1))

cbar = fig.colorbar(
    plt_rvor2, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=rvor_ticks, extend='both',
    anchor=(0.5, 1.5), panchor=(0.5, 0))
# cbar.ax.set_xlabel(
#     'Relative vorticity [$10^{-4}\;s^{-1}$]' + '\nfrom' + \
#     ' ' + str(np.min(time[mask_m]))[0:19] + ' to ' + \
#     str(np.max(time[mask_m]))[0:19], fontsize=10)
# ax.contour(
#     lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
#     linewidths=0.25, linestyles='solid')
ax.contourf(mask_lon2, mask_lat2, masked,
            colors='white', levels=np.array([0.5, 1.5]))
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'), dpi=600)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot 10-meter winds and rvor in sim 2010-08-05 11 UTC

######## ascat bd
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

# mask analysis region outside
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc'
ncfile = xr.open_dataset(file)
lon = ncfile.lon.values
lat = ncfile.lat.values

middle_i = int(lon.shape[1]/2)
lon_m = lon.copy()
lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
lon_s2 = lon_m[:, middle_i:]
lat_s2 = lat[:, middle_i:]
# fil_rvor2 = median_filter(rvor2, 3, )
coors_s2 = np.hstack((lon_s2.reshape(-1, 1), lat_s2.reshape(-1, 1)))
mask_s2 = poly_path.contains_points(coors_s2).reshape(
    lon_s2.shape[0], lon_s2.shape[1])
masked_s2 = np.zeros_like(lon_s2)
masked_s2[mask_s2] = 1
masked_s2[:, 0] = 0
masked_s2[:, -1] = 0

######## mask topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

orig_simulation = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20100805110000.nc')
lon = orig_simulation.lon.values[80:920, 80:920]
lat = orig_simulation.lat.values[80:920, 80:920]
time = orig_simulation.time.values
wind_u = orig_simulation.U_10M.values[0, 80:920, 80:920]
wind_v = orig_simulation.V_10M.values[0, 80:920, 80:920]

wind_speed = (wind_u**2 + wind_v**2)**0.5

# plot
# windlevel = np.arange(0, 18.1, 0.1)
# ticks = np.arange(0, 18.1, 3)

windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

fig, ax = framework_plot1("1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,)
plt_wind1 = ax.pcolormesh(
    lon, lat, wind_speed,
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
ax.text(
    -23, 34, str(time[0])[0:10] + ' ' + str(time[0])[11:13] + ':00 UTC',)
# ax.contourf(
#     mask_lon2, mask_lat2, masked,
#     colors='white', levels=np.array([0.5, 1.5]))
ax.contourf(
    lon, lat, model_topo_mask,
    colors='white', levels=np.array([0.5, 1.5]))
ax.contour(
    lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
cbar = fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("10-meter wind velocity [$m\;s^{-1}$]")
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
fig.savefig(
    'figures/10_validation2obs/10_03_winds/10_03.3.0 Regional winds in sim at 2010080511.png', dpi=600)


dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
rvor = mpcalc.vorticity(
    wind_u * units('m/s'), wind_v * units('m/s'), dx, dy, dim_order='yx',
    ) * 10**4

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="10-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-6, 6.01, 0.05),
        'ticks': np.arange(-6, 6.1, 2),
        'time_point': time[0], 'time_location': [-23, 34], },
)
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
ax.contour(
    lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
fig.savefig(
    'figures/10_validation2obs/10_03_winds/10_03.3.1 Relative vorticity in sim at 2010080511.png', dpi=600,)


'''
'''
# endregion
# =============================================================================

