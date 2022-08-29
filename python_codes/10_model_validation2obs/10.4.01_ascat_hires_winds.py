

# =============================================================================
# region import packages


# basic library
import numpy as np
import xarray as xr
import os
import glob
from scipy.ndimage import median_filter

import sys  # print(sys.path)
sys.path.append(
    '/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
sys.path.append('/scratch/snx3000/qgao')


# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['backend'] = 'Qt4Agg'  #
# mpl.get_backend()

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

from DEoAI_analysis.module.mapplot import (
    framework_plot,
    framework_plot1,
)


# data analysis
import metpy.calc as mpcalc
from metpy.units import units
import dask
dask.config.set({"array.slicing.split_large_chunks": False})

from DEoAI_analysis.module.namelist import (
    transform,
    rvor_level,
    rvor_ticks,
    rvor_cmp,
)



# endregion
# =============================================================================


# =============================================================================
# region download ERA-Interim analysis and forecats

################################
# forecast: 20100805 06 UTC (20100805 00, 6 h), ECMWF_201008050000_00600_GB
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2010-08-05",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "34.128/165.128/166.128",
    "step": "6",
    "stream": "oper",
    "time": "00:00:00",
    "type": "fc",
    "target": "scratch/ascat_hires_winds0/ECMWF_201008050000_00600_GB",
})
################################
# forecast: 20100805 09 UTC (20100805 00, 9 h), ECMWF_201008050000_00900_GB
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2010-08-05",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "34.128/165.128/166.128",
    "step": "9",
    "stream": "oper",
    "time": "00:00:00",
    "type": "fc",
    "target": "scratch/ascat_hires_winds0/ECMWF_201008050000_00900_GB",
})
'''
'''
################################
# forecast: 20100805 12 UTC (20100805 00, 12 h), ECMWF_201008050000_01200_GB
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2010-08-05",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "34.128/165.128/166.128",
    "step": "12",
    "stream": "oper",
    "time": "00:00:00",
    "type": "fc",
    "target": "scratch/ascat_hires_winds0/ECMWF_201008050000_01200_GB",
})

################################
# forecast: 20100805 15 UTC (20100805 0, 15 h), ECMWF_201008050000_01500_GB
# (20100805 12, 3 h), ECMWF_201008051200_00300_GB
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2010-08-05",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "34.128/165.128/166.128",
    "step": "15",
    "stream": "oper",
    "time": "00:00:00",
    "type": "fc",
    "target": "scratch/ascat_hires_winds0/ECMWF_201008050000_01500_GB",
})


'''
import numpy as np
import xarray as xr
ei_201008 = xr.open_dataset(
    'scratch/ascat_hires_winds0/era_interim_201008_0_12_UTC_3_9_12_step_sst_10uv.grib',
    engine = 'cfgrib')

################################

ecmwf_data01 = xr.open_dataset(
    'DEoAI_analysis/fortran/AWDP/awdp/tests/hires/ECMWF_201510271200_00300_GB',
    engine='cfgrib')

ecmwf_data01.time.values
ecmwf_data01.step.values
ecmwf_data01.valid_time.values
time1 = np.datetime64('2010-08-05T00:00')
step1 = np.timedelta64(32400000000000, 'ns')
valid_time1 = np.datetime64('2010-08-05T09:00')

'''
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
# file = 'ascat_20151027_164500_metopa_46811_srv_o_063_ovw.nc'
# output = 'figures/00_test/trial1.png'
# file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc'
# output = 'figures/10_validation2obs/10_06_hires_winds/10_06.0.0 Global 6.25km winds in ascat at 2010080510.png'
file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_057_ovw.nc'
output = 'figures/10_validation2obs/10_06_hires_winds/10_06.0.3 Global 5.7km winds in ascat at 2010080510.png'

ncfile = xr.open_dataset(file)
lon = ncfile.lon.values
lat = ncfile.lat.values
wind_speed = ncfile.wind_speed.values

wind_speed.shape[0] * \
    wind_speed.shape[1] == np.sum(np.isnan(ncfile.wind_speed.values))

# np.sum(np.isnan(ncfile.wind_speed.values))
# np.sum(np.isnan(ncfile.model_speed.values))
# np.sum(np.isnan(ncfile.wind_dir.values))

# ddd = lon[1500, 1:] - lon[1500, :-1]
# ddd[80]
middle_i = int(lon.shape[1]/2)
windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

fig, ax = framework_plot(
    "global", figsize=np.array([17.6, 10.24]) / 2.54, lw=0.1, labelsize=10)
plt_wind1 = ax.pcolormesh(
    lon[:, 0:middle_i], lat[:, 0:middle_i],
    wind_speed[:, 0:middle_i][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
plt_wind2 = ax.pcolormesh(
    lon[:, middle_i:], lat[:, middle_i:],
    wind_speed[:, middle_i:][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
ax.contour(
    lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
    linewidths=0.25, linestyles='solid')

cbar = fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06,
    shrink=0.5, aspect=30, ticks=ticks, extend='max',)
ax.text(
    0, -136,
    'Wind velocity [$m \; s^{-1}$]' + ' from ' + str(ncfile.start_date) +
    ' ' + str(ncfile.start_time) + ' to ' +
    str(ncfile.stop_date) + ' ' + str(ncfile.stop_time),
    horizontalalignment='center')
fig.subplots_adjust(left=0.05, right=0.975, bottom=0.10, top=0.98)
fig.savefig(output, dpi=600)



'''
import xarray as xr
ecmwf_data00 = xr.open_dataset(
    'DEoAI_analysis/fortran/AWDP/awdp/tests/hires/ECMWF_201510271200_00000_GB',
    engine = 'cfgrib')
ecmwf_data01 = xr.open_dataset(
    'DEoAI_analysis/fortran/AWDP/awdp/tests/hires/ECMWF_201510271200_00300_GB',
    engine = 'cfgrib')
ecmwf_data02 = xr.open_dataset(
    'DEoAI_analysis/fortran/AWDP/awdp/tests/hires/ECMWF_201510271200_00600_GB',
    engine = 'cfgrib')
ecmwf_data03 = xr.open_dataset(
    'DEoAI_analysis/fortran/AWDP/awdp/tests/hires/ECMWF_201510271200_00900_GB',
    engine = 'cfgrib')
(ecmwf_data01.v10 == ecmwf_data02.v10)

ei_data01 = xr.open_dataset(
    'scratch/ascat_hires_winds0/ECMWF_201008050000_00900_GB',
    engine='cfgrib')
ei_data02 = xr.open_dataset(
    'scratch/ascat_hires_winds0/ECMWF_201008050000_01200_GB',
    engine='cfgrib')
ei_data03 = xr.open_dataset(
    'scratch/ascat_hires_winds0/ECMWF_201008051200_00300_GB',
    engine='cfgrib')


ecmwf_data11 = xr.open_dataset(
    'scratch/ascat_hires_winds0/era5_var_20100805_09.grib',
    engine='cfgrib')
ecmwf_data12 = xr.open_dataset(
    'scratch/ascat_hires_winds0/era5_var_20100805_12.grib',
    engine='cfgrib')
ecmwf_data13 = xr.open_dataset(
    'scratch/ascat_hires_winds0/era5_var_20100805_15.grib',
    engine='cfgrib')


ncfile1 = xr.open_dataset(
    'scratch/ascat_hires_winds0/ascat_A_20151027_164500_250_test.nc',)


'''
# endregion
# =============================================================================


# =============================================================================
# region plot regional ascat winds around madeira

file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc'
output = 'figures/10_validation2obs/10_06_hires_winds/10_06.0.1 Regional 6.25km winds in ascat at 2010080510.png'
# file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_057_ovw.nc'
# output = 'figures/10_validation2obs/10_06_hires_winds/10_06.0.4 Regional 5.7km winds in ascat at 2010080510.png'

#### create a boundary to mask outside
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

ncfile = xr.open_dataset(file)
lon = ncfile.lon.values
lat = ncfile.lat.values
wind_speed = ncfile.wind_speed.values
time = ncfile.time.values

#### madaira time
middle_i = int(lon.shape[1]/2)
lon_m = lon.copy()
lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
coors_m = np.hstack((lon_m.reshape(-1, 1), lat.reshape(-1, 1)))
mask_m = poly_path.contains_points(coors_m).reshape(
    lon.shape[0], lon.shape[1])
# time[mask_m]

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

windlevel = np.arange(0, 12.1, 0.1)
ticks = np.arange(0, 12.1, 2)

fig, ax = framework_plot1("1km_lb", figsize=np.array([8.8, 9.3]) / 2.54,)
plt_wind1 = ax.pcolormesh(
    lon[:, 0:middle_i], lat[:, 0:middle_i],
    wind_speed[:, 0:middle_i][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
plt_wind2 = ax.pcolormesh(
    lon[:, middle_i:], lat[:, middle_i:],
    wind_speed[:, middle_i:][:-1, :-1],
    cmap=cm.get_cmap('viridis', len(windlevel)), transform=transform,
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),)
# ax.contour(
#     lon1, lat1, analysis_region, colors='red', levels=np.array([0.5]),
#     linewidths=0.25, linestyles='solid')
ax.contourf(mask_lon2, mask_lat2, masked,
            colors='white', levels=np.array([0.5, 1.5]))
ax.contour(
    lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
cbar = fig.colorbar(
    plt_wind1, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    '10-meter wind velocity [$m \; s^{-1}$] by ASCAT' + '\nfrom ' +
    str(np.min(time[mask_m]))[0:10] + \
    ' ' + str(np.min(time[mask_m]))[11:16] + ' to ' + \
    str(np.max(time[mask_m]))[11:16] + ' UTC', fontsize=10)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)
fig.savefig(output, dpi=600)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region calculate and plot relative vorticity

#### create a boundary to mask outside
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

#### input ascat winds
# file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc'
# output = 'figures/10_validation2obs/10_06_hires_winds/10_06.0.2 Regional 6.25km relative vorticity in ascat at 2010080510.png'
file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_057_ovw.nc'
output = 'figures/10_validation2obs/10_06_hires_winds/10_06.0.5 Regional 5.7km relative vorticity in ascat at 2010080510.png'

ncfile = xr.open_dataset(file)
lon = ncfile.lon.values
lat = ncfile.lat.values
wind_speed = ncfile.wind_speed.values
wind_dir = ncfile.wind_dir.values
time = ncfile.time.values

#### madaira time
middle_i = int(lon.shape[1]/2)
lon_m = lon.copy()
lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
coors_m = np.hstack((lon_m.reshape(-1, 1), lat.reshape(-1, 1)))
mask_m = poly_path.contains_points(coors_m).reshape(
    lon_m.shape[0], lon_m.shape[1])
# time[mask_m]

#### calculate relative vorticity

# lon_s1 = lon_m[:, 0:middle_i]
# lat_s1 = lat[:, 0:middle_i]
# wind_speed_s1 = wind_speed[:, 0:middle_i]
# wind_dir_s1 = wind_dir[:, 0:middle_i]
# wind_u1 = wind_speed_s1 * np.sin(np.deg2rad(wind_dir_s1))
# wind_v1 = wind_speed_s1 * np.cos(np.deg2rad(wind_dir_s1))
# dx1, dy1 = mpcalc.lat_lon_grid_deltas(lon_s1, lat_s1)
# rvor1 = mpcalc.vorticity(
#     wind_u1 * units('m/s'), wind_v1 * units('m/s'),
#     dx1, dy1, dim_order='yx') * 10**4
# # fil_rvor1 = median_filter(rvor1, 3, )
# coors_s1 = np.hstack((lon_s1.reshape(-1, 1), lat_s1.reshape(-1, 1)))
# mask_s1 = poly_path.contains_points(coors_s1).reshape(
#     lon_s1.shape[0], lon_s1.shape[1])
# masked_s1 = np.zeros_like(lon_s1)
# masked_s1[mask_s1] = 1

lon_s2 = lon_m[:, middle_i:]
lat_s2 = lat[:, middle_i:]
wind_speed_s2 = wind_speed[:, middle_i:]
wind_dir_s2 = wind_dir[:, middle_i:]
wind_u2 = wind_speed_s2 * np.sin(np.deg2rad(wind_dir_s2))
wind_v2 = wind_speed_s2 * np.cos(np.deg2rad(wind_dir_s2))
dx2, dy2 = mpcalc.lat_lon_grid_deltas(lon_s2, lat_s2)
rvor2 = mpcalc.vorticity(
    wind_u2 * units('m/s'), wind_v2 * units('m/s'),
    dx2, dy2, dim_order='yx') * 10**4
# fil_rvor2 = median_filter(rvor2, 3, )
coors_s2 = np.hstack((lon_s2.reshape(-1, 1), lat_s2.reshape(-1, 1)))
mask_s2 = poly_path.contains_points(coors_s2).reshape(
    lon_s2.shape[0], lon_s2.shape[1])
masked_s2 = np.zeros_like(lon_s2)
masked_s2[mask_s2] = 1
masked_s2[:, 0] = 0
masked_s2[:, -1] = 0

# cs_s1 = plt.contour(lon_s2, lat_s2, masked_s2, levels=np.array([0.5]),)
# poly_path_s1 = cs_s1.collections[0].get_paths()[0]
# mask_s2_2 = poly_path_s1.contains_points(coors).reshape(
#     mask_lon2.shape[0], mask_lon2.shape[1])
# masked_s2_2 = np.zeros_like(mask_lon2)
# masked_s2_2[mask_s2_2] = 1

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

# plt_rvor1 = ax.pcolormesh(
#     lon_s1, lat_s1, rvor1, cmap=rvor_cmp,
#     rasterized=True, transform=transform, zorder=-2,
#     norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),)
plt_rvor2 = ax.pcolormesh(
    lon_s2, lat_s2, rvor2, cmap=rvor_cmp, rasterized=True,
    transform=transform, zorder=-2,
    norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),)

cbar = fig.colorbar(
    plt_rvor2, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=rvor_ticks, extend='both',
    anchor=(0.5, 1.5), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    '10-meter relative vorticity [$10^{-4}\;s^{-1}$] by ASCAT' + '\nfrom' + \
    ' ' + str(np.min(time[mask_m]))[0:10] + \
    ' ' + str(np.min(time[mask_m]))[11:16] + ' to ' + \
    str(np.max(time[mask_m]))[11:16] + ' UTC', fontsize=10)

ax.contourf(mask_lon2, mask_lat2, masked,
            colors='white', levels=np.array([0.5, 1.5]))
# ax.contour(
#     lon_s1, lat_s1, masked_s1,
#     colors='red', levels=np.array([0.5]),
#     linewidths=0.25, linestyles='solid')
ax.contour(
    lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)
fig.savefig(output, dpi=600)


'''
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

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

ax.contour(
    lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
'''
# endregion
# =============================================================================

