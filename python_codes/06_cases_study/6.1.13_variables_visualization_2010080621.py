

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


timepoint = 73
# new 3d simulation
# =============================================================================
# region load data


# rvor
rvorticity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/rvorticity_3d_20100801_09z.nc')
time = rvorticity_3d_20100801_09z.time.values
lon = rvorticity_3d_20100801_09z.lon.values
lat = rvorticity_3d_20100801_09z.lat.values
rvor = rvorticity_3d_20100801_09z.relative_vorticity[
    timepoint + 68, 1, :, :].values * 10**4

# theta
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
theta100 = theta_3d_20100801_09z.theta[timepoint + 68, 1, :, :].values
theta10 = theta_3d_20100801_09z.theta[timepoint + 68, 0, :, :].values

# wind
velocity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/velocity_3d_20100801_09z.nc')
wind_u = velocity_3d_20100801_09z.u_earth[timepoint+68, 1, :, :].values
wind_v = velocity_3d_20100801_09z.v_earth[timepoint+68, 1, :, :].values

# new simulation file
nc3d_lb_m = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100806210000.nc')
nc3d_lb_z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100806210000z.nc')

# new simulation constant file
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan


################################ others
######## parameter settings
grid_size = 1.2  # in km^2
median_filter_size = 3
maximum_filter_size = 50
min_rvor = 3.
min_max_rvor = 4.
min_size = 100.
min_size_theta = 450.
min_size_dir = 450
min_size_dir1 = 900
max_dir = 30
max_dir1 = 40
max_dir2 = 50
max_distance2radius = 5
reject_info = True

from matplotlib.path import Path
polygon=[
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
    ]
poly_path=Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)


################################ original identified
# original_identified_rvor = tb.open_file(
#     "scratch/rvorticity/rvor_identify/identified_transformed_rvor_20100803_09_002_rejected.h5",
#     mode="r")
# original_exp = original_identified_rvor.root.exp1


################################ wavelet transform
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices1, is_vortex1, vortices_count1, vortex_indices1, theta_anomalies1,
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta10, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)


# endregion
# =============================================================================


# =============================================================================
# region relative vorticity with cross section

# add cross section line1
upstream_length = 1.2
downstream_length = 3
lineangle1 = 75
startpoint1 = [
    center_madeira[1] + upstream_length * np.sin(np.deg2rad(lineangle1 + 0)),
    center_madeira[0] + upstream_length * np.cos(np.deg2rad(lineangle1 + 0)),
]
endpoint1 = [
    center_madeira[1] - downstream_length * np.sin(np.deg2rad(lineangle1 + 0)),
    center_madeira[0] - downstream_length * np.cos(np.deg2rad(lineangle1 + 0)),
]

# add cross section line2
lineangle2 = 57
startpoint2 = [
    center_madeira[1] + upstream_length * np.sin(np.deg2rad(lineangle2 + 0)),
    center_madeira[0] - 0.1 + upstream_length * np.cos(
        np.deg2rad(lineangle2 + 0)),
]
endpoint2 = [
    center_madeira[1] - downstream_length * np.sin(np.deg2rad(lineangle2 + 0)),
    center_madeira[0] - 0.1 - downstream_length * np.cos(
        np.deg2rad(lineangle2 + 0)),
]


fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    # vorticity_elements={
    #     'rvor': rvor,
    #     'lon': lon,
    #     'lat': lat,
    #     'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
    #     'time_point': time[timepoint + 68], 'time_location': [-23, 34], },
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': '           ', 'time_location': [-23, 34], },
    gridlines=False,
)
# add cross section line1
line1, = ax.plot(
    [startpoint1[1], endpoint1[1]], [startpoint1[0], endpoint1[0]],
    lw=0.5, linestyle="--", color='black', zorder=2)

# add cross section line2
line2, = ax.plot(
    [startpoint2[1], endpoint2[1]], [startpoint2[0], endpoint2[0]],
    lw=0.5, linestyle="--", color='black', zorder=2)

ax.text(startpoint1[1] - 0.5, startpoint1[0] + 0.1, '$C_1$')
ax.text(endpoint1[1] + 0.1, endpoint1[0], '$C_2$')
ax.text(startpoint2[1], startpoint2[0] + 0.1, '$C_3$')
ax.text(endpoint2[1] + 0.1, endpoint2[0] - 0.3, '$C_4$')

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
# fig.savefig(
#     'figures/06_case_study/06_02_time_points_new/6.3.0 relative vorticity with cross section in 2010080621.png')
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6.3.0.1 relative vorticity with cross section in 2010080621_nogrid.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint + 68], 'time_location': [-23, 34], },
    )
ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.0 rvor_2010080621.png')


# endregion
# =============================================================================


# =============================================================================
# region plot theta

# time[timepoint + 68]

# stats.describe(theta.flatten()) # [280, 305]
theta_min = 281 + 9
theta_mid = 293
theta_max = 305 - 9

theta_ticks = np.arange(theta_min, theta_max + 0.01, 1)
theta_level = np.arange(theta_min, theta_max + 0.01, 0.025)

fig, ax = framework_plot1("1km_lb",)

theta_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + ' ' + \
    str(time[timepoint + 68])[11:13] + ':00 UTC')
# plt_theta = ax.pcolormesh(
#     lon, lat, theta10, cmap=rvor_cmp, rasterized=True, transform=transform,
#     norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )
plt_theta = ax.pcolormesh(
    lon, lat, theta100, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_theta, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=theta_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
# cbar.ax.set_xlabel("10-meter potential temperature [K]")
cbar.ax.set_xlabel("100-meter potential temperature [K]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
# fig.savefig(
#     'figures/06_case_study/06_02_time_points_new/6_1_0.1 theta_2010080621.png')
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.11 theta100_2010080621.png')


'''
# plt_theta = ax.pcolormesh(
#     lon, lat, theta, cmap='terrain', rasterized=True, transform=transform,
#     norm=BoundaryNorm(theta_level, ncolors=len(theta_level), clip=True), )

plt_theta = ax.contour(
    lon, lat, theta,
    colors='black', levels=theta_ticks, linewidths=0.5
    )
ax.clabel(plt_theta, inline=1, fontsize=10, colors='black',
          fmt='%1.0f', levels=theta_ticks[::2], inline_spacing=10,
          )
'''
# endregion
# =============================================================================


# =============================================================================
# region plot relative humidity

relhum = nc3d_lb_z.RELHUM[0, 1, :, :].values
# stats.describe(relhum.flatten()) # [4, 100]
relhum_min = 86 - 12
relhum_mid = 86
relhum_max = 86 + 12

relhum_ticks = np.arange(relhum_min, relhum_max + 0.01, 4)
relhum_level = np.arange(relhum_min, relhum_max + 0.01, 0.1)

fig, ax = framework_plot1(
    "1km_lb",
)

relhum_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + \
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
plt_relhum = ax.pcolormesh(
    lon, lat, relhum, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(relhum_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_relhum, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=relhum_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter relative humidity [%]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.2 relhum_2010080621.png')



'''
nc3d_lb = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100806210000.nc')
nc1h = xr.open_dataset('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20051101000000.nc')
nc1h_second = xr.open_dataset('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20051101000000.nc')

nc1h.PS
nc1h.HPBL
'''
# endregion
# =============================================================================


# =============================================================================
# region plot temperature

tem = nc3d_lb_z.T[0, 1, :, :].values
# stats.describe(tem.flatten()) # [4, 100]
tem_min = 281 + 9
tem_mid = 293
tem_max = 305 - 9

tem_ticks = np.arange(tem_min, tem_max + 0.01, 1)
tem_level = np.arange(tem_min, tem_max + 0.01, 0.025)

fig, ax = framework_plot1(
    "1km_lb",
)

tem_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + \
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
plt_tem = ax.pcolormesh(
    lon, lat, tem, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(tem_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_tem, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=tem_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter temperature [K]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.3 tem_2010080621.png')

'''
'''

# endregion
# =============================================================================


# =============================================================================
# region plot pressure

pp = nc3d_lb_z.PP[0, 1, :, :].values
p0 = p0sl * np.exp(-(g * m * 100.0 / (r0 * t0sl)))
pres = (pp + p0)/100
# stats.describe(pres.flatten()) # [4, 100]
pres_min = 1003 - 6
pres_mid = 1003
pres_max = 1003 + 6

pres_ticks = np.arange(pres_min, pres_max + 0.01, 2)
pres_level = np.arange(pres_min, pres_max + 0.01, 0.08)

fig, ax = framework_plot1("1km_lb",)

pres_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + \
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
plt_pres = ax.pcolormesh(
    lon, lat, pres, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(pres_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_pres, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=pres_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter pressure [hPa]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.4 pres_2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region plot specific humidity
pp = nc3d_lb_z.PP[0, 1, :, :].values
p0 = p0sl * np.exp(-(g * m * 100.0 / (r0 * t0sl)))
pres = pp + p0
tem = nc3d_lb_z.T[0, 1, :, :].values
relhum = nc3d_lb_z.RELHUM[0, 1, :, :].values

omega = mpcalc.saturation_mixing_ratio(pres * units('Pa'), tem * units('K'))
spec_hum = relhum/100 * omega / (1 + relhum/100 * omega)
# stats.describe(spec_hum.flatten()) # [0.07, 100]
spechum_min = 0.012 - 0.003
spechum_mid = 0.012
spechum_max = 0.012 + 0.003

spechum_ticks = np.arange(spechum_min, spechum_max + 0.0001, 0.001)
spechum_level = np.arange(spechum_min, spechum_max + 0.00001, 0.00003)

cbar_top = cm.get_cmap('Blues_r', int(np.floor(len(spechum_level) / 2)))
cbar_bottom = cm.get_cmap('Reds', int(np.floor(len(spechum_level) / 2)))
cbar_colors = np.vstack(
    (cbar_top(np.linspace(0, 1, int(np.floor(len(spechum_level) / 2)))),
     [1, 1, 1, 1],
     cbar_bottom(np.linspace(0, 1, int(np.floor(len(spechum_level) / 2))))))
cbar_cmp = ListedColormap(cbar_colors, name='RedsBlues_r')

fig, ax = framework_plot1("1km_lb",)

spechum_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + \
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
plt_spechum = ax.pcolormesh(
    lon, lat, spec_hum, cmap=cbar_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(spechum_level, ncolors=cbar_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_spechum, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=spechum_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter specific humidity [-]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.5 spechum_2010080621.png')

'''
mix_ratio = mpcalc.mixing_ratio_from_relative_humidity(
    relhum, tem * units('K'), pres * units('Pa'))
spec_hum2 = mpcalc.specific_humidity_from_mixing_ratio(mix_ratio)

rec_relhum = mpcalc.relative_humidity_from_specific_humidity(
    spec_hum1, tem * units('K'), pres * units('Pa'))

# check
specific_hum = xr.open_dataset(
    'download.nc'
)
'''
# endregion
# =============================================================================


# =============================================================================
# region plot wind
# wind_earth = xr.open_dataset(
#     'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m201008.nc')
# wind_earth_1h_100m = xr.open_dataset(
#     'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')
# time1 = wind_earth_1h_100m.time.values
# istart = 5156 + timepoint
# time1[istart]
# wind = wind_earth_1h_100m.strength[istart, :, :].values

wind = (wind_u**2 + wind_v**2)**0.5
windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)

fig, ax = framework_plot1("1km_lb",)

plt_wind = ax.pcolormesh(
    lon, lat, wind, cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, rasterized=True,
    # zorder=-2,
    )
cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter wind velocity [$m\;s^{-1}$]")
iarrow = 30
ax.quiver(
    lon[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
    lat[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
    wind_u[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow] /
    (wind[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow]),
    wind_v[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow] /
    (wind[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow]),
    # wind[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
    # cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    # norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    # scale=80,
    # headlength=20,
    # headwidth=12,
    # width=0.004,
    alpha=0.5,
    )
wind_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] +
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC',
    )

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.6 wind_2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region plot potential vorticity
filelist_pl = sorted(glob.glob(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd201008*p.nc'))
# ifinal = 2
plev_3d_sim = xr.open_mfdataset(
    filelist_pl[timepoint + 68], concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
).metpy.parse_cf()

wind_u = plev_3d_sim.U.values
wind_v = plev_3d_sim.V.values
tem = plev_3d_sim.T.values
pres = plev_3d_sim.pressure.values
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
# time2 = plev_3d_sim.time.values[0]

theta = tem * (p0sl/pres[None, :, None, None])**(r/cp)
potential_vorticity = mpcalc.potential_vorticity_baroclinic(
    theta[0] * units('K'), pres[:, None, None] * units('Pa'),
    wind_u[0] * units('m/s'), wind_v[0] * units('m/s'),
    dx[None, :, :], dy[None, :, :], np.deg2rad(lat),
)
# stats.describe((potential_vorticity[-1] * 10**6).flatten())
pv_min = -12
pv_mid = 0
pv_max = 12

pv_ticks = np.arange(pv_min, pv_max + 0.01, 3)
pv_level = np.arange(pv_min, pv_max + 0.01, 0.1)

fig, ax = framework_plot1("1km_lb",)

pv_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + \
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
plt_pv = ax.pcolormesh(
    lon, lat, potential_vorticity[-2] * 10**6, cmap=rvor_cmp,
    rasterized=True, transform=transform,
    norm=BoundaryNorm(pv_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_pv, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=pv_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Potential vorticity at 950 hPa [pvu $10^{-6} \; K m^{2} kg^{-1} s ^{-1}$]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.7 pv_2010080621.png')




# endregion
# =============================================================================


# =============================================================================
# region plot inversion base height

inversion_height = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc')

hinv = inversion_height.inversion_height[timepoint + 68, :, :].values
# stats.describe(hinv.flatten(), nan_policy = 'omit')
hinv_min = 50
hinv_mid = 500
hinv_max = 950

hinv_ticks = np.arange(hinv_min, hinv_max + 0.01, 150)
hinv_level = np.arange(hinv_min, hinv_max + 0.01, 4)

fig, ax = framework_plot1("1km_lb",)

hinv_time = ax.text(
    -23, 34, str(time[timepoint + 68])[0:10] + \
    ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
plt_hinv = ax.pcolormesh(
    lon, lat, hinv, cmap=rvor_cmp,
    rasterized=True, transform=transform,
    norm=BoundaryNorm(hinv_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_hinv, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=hinv_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Inversion height [m]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.8 hinv_2010080621.png')


# endregion
# =============================================================================


# =============================================================================
# region plot PBL height in model

# pblh = nc3d_lb_m.HPBL[0, 80:920, 80:920].values
# # stats.describe(pblh.flatten(), nan_policy = 'omit')
# pblh_min = 50
# pblh_mid = 500
# pblh_max = 950

# pblh_ticks = np.arange(pblh_min, pblh_max + 0.01, 150)
# pblh_level = np.arange(pblh_min, pblh_max + 0.01, 4)

# fig, ax = framework_plot1(
#     "1km_lb",
# )

# pblh_time = ax.text(
#     -23, 34, str(time[timepoint + 68])[0:10] +
#     ' ' + str(time[timepoint + 68])[11:13] + ':00 UTC')
# plt_pblh = ax.pcolormesh(
#     lon, lat, pblh, cmap=rvor_cmp,
#     rasterized=True, transform=transform,
#     norm=BoundaryNorm(pblh_level, ncolors=rvor_cmp.N, clip=False), )
# cbar = fig.colorbar(
#     plt_pblh, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
#     shrink=1, aspect=25, ticks=pblh_ticks, extend='both',
#     anchor=(0.5, 1), panchor=(0.5, 0))
# cbar.ax.set_xlabel(
#     "Planetary boundary layer height [m]")

# ax.contour(lon, lat, is_vortex1,
#            colors='lime', levels=np.array([-0.5, 0.5]),
#            linewidths=0.5, linestyles='solid'
#            )
# fig.savefig(
#     'figures/06_case_study/06_02_time_points_new/6_1_0.9 pblh_2010080621.png')


# endregion
# =============================================================================


# =============================================================================
# region plot 10m relative vorticity

rvor10 = rvorticity_3d_20100801_09z.relative_vorticity[
    timepoint + 68, 0, :, :].values * 10**4

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="10-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor10,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint + 68], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
# ax.contour(lon, lat, experiment2.vortex_info.cols.is_vortex[timepoint],
#            colors='m', levels=np.array([-0.5, 0.5]),
#            linewidths=0.5, linestyles='solid'
#            )
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_1_0.10 10mrvor_2010080621.png')

# endregion
# =============================================================================


# old simulation
# =============================================================================
# =============================================================================
# region load data

# rvor
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201008.nc')
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values

# wind
iyear = 4
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

# old simulation constant file
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')))

dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

# parameter
grid_size = 1.2  # in km^2
median_filter_size = 3
maximum_filter_size = 50
min_rvor = 3.
min_max_rvor = 4.
min_size = 100.
min_size_theta = 450.
min_size_dir = 450
min_size_dir1 = 900
max_dir = 30
max_dir1 = 40
max_dir2 = 50
max_distance2radius = 5
reject_info = True

# create mask
from matplotlib.path import Path
polygon=[
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
    ]
poly_path=Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)

# import timepoint data
i = np.where(
    wind_100m.time == np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values

orig_simulation2nd = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20100806210000.nc')
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)


# endregion
# =============================================================================


# =============================================================================
# region plot rvor

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.0 old_sim rvor_2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region plot 2m temperature

tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
# stats.describe(tem2m.flatten()) # [4, 100]
tem_min = 281 + 9
tem_mid = 293
tem_max = 305 - 9

tem_ticks = np.arange(tem_min, tem_max + 0.01, 1)
tem_level = np.arange(tem_min, tem_max + 0.01, 0.025)

fig, ax = framework_plot1("1km_lb",)

tem_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')
plt_tem = ax.pcolormesh(
    lon, lat, tem2m, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(tem_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_tem, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=tem_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("2m Temperature [K]")
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.1 old_sim 2mtem_2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region plot surface pressure

pres = orig_simulation.PS[0, 80:920, 80:920].values / 100
# stats.describe(pres.flatten()) # [4, 100]
pres_mid = 1013
pres_min = pres_mid - 6
pres_max = pres_mid + 6

pres_ticks = np.arange(pres_min, pres_max + 0.01, 2)
pres_level = np.arange(pres_min, pres_max + 0.01, 0.08)

fig, ax = framework_plot1("1km_lb",)

pres_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')
plt_pres = ax.pcolormesh(
    lon, lat, pres, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(pres_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_pres, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=pres_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Surface Pressure [hPa]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.2 old_sim surface_pressure_2010080621.png')


# endregion
# =============================================================================


# =============================================================================
# region plot theta

# stats.describe(theta.flatten()) # [280, 305]
theta_min = 281 + 9
theta_mid = 293
theta_max = 305 - 9

theta_ticks = np.arange(theta_min, theta_max + 0.01, 1)
theta_level = np.arange(theta_min, theta_max + 0.01, 0.025)

fig, ax = framework_plot1("1km_lb",)

theta_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + ' ' + \
    str(time[timepoint])[11:13] + ':00 UTC')
plt_theta = ax.pcolormesh(
    lon, lat, theta, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_theta, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=theta_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Surface potential temperature [K]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.3 old_sim surface theta_2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region plot wind

windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)
wind = (wind_u**2 + wind_v**2)**0.5

fig, ax = framework_plot1("1km_lb",)

plt_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + ' ' + \
    str(time[timepoint])[11:13] + ':00 UTC')
plt_wind = ax.pcolormesh(
    lon, lat, wind, cmap=cm.get_cmap('viridis', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, rasterized=True, zorder=-2,)

cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter wind velocity [$m\;s^{-1}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.11 old_sim 100_m_winds_2010080621.png')

# endregion
# =============================================================================


# radiation
# =============================================================================
# region plot averaged surface sensible heat flux ASHFL_S

variables = orig_simulation2nd.ASHFL_S[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_mid = 0
cbar_min = cbar_mid - 40
cbar_max = cbar_mid + 40
cbar_ticks = np.arange(cbar_min, cbar_max + 0.01, 10)
cbar_level = np.arange(cbar_min, cbar_max + 0.01, 0.5)

cbar_top = cm.get_cmap('Blues_r', int(np.floor(len(cbar_level) / 2)))
cbar_bottom = cm.get_cmap('Reds', int(np.floor(len(cbar_level) / 2)))
cbar_colors = np.vstack(
    (cbar_top(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2)))),
     [1, 1, 1, 1],
     cbar_bottom(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2))))))
cbar_cmp = ListedColormap(cbar_colors, name='RedsBlues_r')

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cbar_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Averaged surface sensible heat flux [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.4 old_sim averaged surface_sensible_heat_flux_2010080621.png')

'''
'ASHFL_S': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    standard_name:  surface_downward_sensible_heat_flux
    long_name:      averaged surface sensible heat flux
    units:          W m-2
    grid_mapping:   rotated_pole
    cell_methods:   time: mean,
'''

# endregion
# =============================================================================


# =============================================================================
# region plot averaged surface latent heat flux ALHFL_S

variables = orig_simulation2nd.ALHFL_S[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_level = np.arange(-180, 0.01, 0.5)
cbar_ticks = np.arange(-180, 0.01, 30)


fig, ax = framework_plot1("1km_lb",)

plt_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('viridis_r', len(cbar_level)),
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='min',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Averaged surface latent heat flux [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.5 old_sim averaged surface_latent_heat_flux_2010080621.png')

'''
'ALHFL_S': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    standard_name:  surface_downward_latent_heat_flux
    long_name:      averaged surface latent heat flux
    units:          W m-2
    grid_mapping:   rotated_pole
    cell_methods:   time: mean,

cbar_mid = -90
cbar_min = cbar_mid - 90
cbar_max = cbar_mid + 90
cbar_ticks = np.arange(cbar_min, cbar_max + 0.01, 30)
cbar_level = np.arange(cbar_min, cbar_max + 0.01, 0.5)

cbar_top = cm.get_cmap('Blues_r', int(np.floor(len(cbar_level) / 2)))
cbar_bottom = cm.get_cmap('Reds', int(np.floor(len(cbar_level) / 2)))
cbar_colors = np.vstack(
    (cbar_top(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2)))),
     [1, 1, 1, 1],
     cbar_bottom(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2))))))
cbar_cmp = ListedColormap(cbar_colors, name='RedsBlues_r')

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cbar_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Averaged surface latent heat flux [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.5 old_sim averaged surface_latent_heat_flux_2010080621.png')

'''

# endregion
# =============================================================================


# =============================================================================
# region plot averaged surface net downward shortwave radiation ASOB_S

variables = orig_simulation2nd.ASOB_S[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_level = np.arange(0, 10.01, 0.05)
cbar_ticks = np.arange(0, 10.1, 2)


fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('RdYlBu_r', len(cbar_level)),
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Averaged surface net downward SW radiation [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.6 old_sim averaged_surface_net_downward_shortwave_radiation_2010080621.png')

'''
'ASOB_S': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    standard_name:  surface_net_downward_shortwave_flux
    long_name:      averaged surface net downward shortwave radiation
    units:          W m-2
    grid_mapping:   rotated_pole
    cell_methods:   time: mean,
'''

# endregion
# =============================================================================


# =============================================================================
# region plot averaged TOA net downward shortwave radiation ASOB_T

variables = orig_simulation2nd.ASOB_T[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_level = np.arange(0, 10.01, 0.05)
cbar_ticks = np.arange(0, 10.1, 2)

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] +
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('RdYlBu_r', len(cbar_level)),
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Averaged TOA net downward SW radiation [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.7 old_sim averaged_TOA_net_downward_shortwave_radiation_2010080621.png')

'''
'ASOB_T': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    standard_name:  net_downward_shortwave_flux_in_air
    long_name:      averaged TOA net downward shortwave radiation
    units:          W m-2
    grid_mapping:   rotated_pole
    cell_methods:   time: mean,
'''

# endregion
# =============================================================================


# =============================================================================
# region plot averaged TOA outgoing longwave radiation ATHB_T

# variables = np.abs(orig_simulation2nd.ATHB_T[0, 80:920, 80:920].values)
variables = orig_simulation2nd.ATHB_T[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_level = np.arange(-310, -284.99, 0.1)
cbar_ticks = np.arange(-310, -284.99, 5)

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('viridis_r', len(cbar_level)),
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Averaged TOA outgoing LW radiation [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.8 old_sim averaged_TOA_outgoing_longwave_radiation_2010080621.png')

'''
'ATHB_T': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    standard_name:  net_downward_longwave_flux_in_air
    long_name:      averaged TOA outgoing longwave radiation
    units:          W m-2
    grid_mapping:   rotated_pole
    cell_methods:   time: mean,
'''

# endregion
# =============================================================================


# =============================================================================
# region plot averaged surface net downward LW radiation ATHB_S

# variables = np.abs(orig_simulation2nd.ATHB_S[0, 80:920, 80:920].values)
variables = orig_simulation2nd.ATHB_S[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_level = np.arange(-100, 0.01, 0.2)
cbar_ticks = np.arange(-100, 0.01, 20)

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('viridis_r', len(cbar_level)),
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='min',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Averaged surface net downward LW radiation [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.9 old_sim averaged_surface_net_downward_longwave_radiation_2010080621.png')

'''
, cmap=cm.get_cmap('viridis_r', len(cbar_level))
'ATHB_S': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    standard_name:  surface_net_downward_longwave_flux
    long_name:      averaged surface net downward longwave radiation
    units:          W m-2
    grid_mapping:   rotated_pole
    cell_methods:   time: mean,
'''

# endregion
# =============================================================================


# =============================================================================
# region plot total downward sw radiation at the surface ASWD_S

# variables = np.abs(orig_simulation2nd.ASWD_S[0, 80:920, 80:920].values)
# # stats.describe(variables.flatten())

# cbar_level = np.arange(0, 100.01, 0.1)
# cbar_ticks = np.arange(0, 100.01, 10)

# fig, ax = framework_plot1("1km_lb",)

# timeplot = ax.text(
#     -23, 34, str(time[timepoint])[0:10] + \
#     ' ' + str(time[timepoint])[11:13] + ':00 UTC')

# plt_variables = ax.pcolormesh(
#     lon, lat, variables, cmap=cm.get_cmap('RdYlBu_r', len(cbar_level)),
#     rasterized=True, transform=transform,
#     norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
# cbar = fig.colorbar(
#     plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
#     shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
#     anchor=(0.5, 1), panchor=(0.5, 0))
# cbar.ax.set_xlabel(
#     "Averaged surface net downward LW radiation [$W\;m^{-2}$]")

# ax.contour(lon, lat, is_vortex_trs,
#            colors='lime', levels=np.array([-0.5, 0.5]),
#            linewidths=0.5, linestyles='solid')
# ax.contourf(lon, lat, model_topo_mask,
#             colors='white', levels=np.array([0.5, 1.5]))

# fig.savefig(
#     'figures/06_case_study/06_02_time_points_new/6_2_1.9 old_sim averaged_surface_net_downward_longwave_radiation_2010080621.png')

'''
'ASWD_S': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    long_name:     total downward sw radiation at the surface
    units:         W m-2
    grid_mapping:  rotated_pole
    cell_methods:  time: mean,
'''

# endregion
# =============================================================================


# =============================================================================
# region plot upward lw radiation at the surface LWU_S

variables = orig_simulation2nd.LWU_S[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

cbar_mid = 430
cbar_min = cbar_mid - 15
cbar_max = cbar_mid + 15
cbar_level = np.arange(cbar_min, cbar_max + 0.01, 0.1)
cbar_ticks = np.arange(cbar_min, cbar_max + 0.01, 5)

cbar_top = cm.get_cmap('Blues_r', int(np.floor(len(cbar_level) / 2)))
cbar_bottom = cm.get_cmap('Reds', int(np.floor(len(cbar_level) / 2)))
cbar_colors = np.vstack(
    (cbar_top(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2)))),
     [1, 1, 1, 1],
     cbar_bottom(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2))))))
cbar_cmp = ListedColormap(cbar_colors, name='RedsBlues_r')

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cbar_cmp,
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Upward lw radiation at the surface [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.10 old_sim upward_lw_radiation_at_the_surface_2010080621.png')

'''
'LWU_S': <xarray.Variable (time: 1, rlat: 1000, rlon: 1000)>
[1000000 values with dtype=float32]
Attributes:
    long_name:     upward lw radiation at the surface
    units:         W m-2
    grid_mapping:  rotated_pole})

'''

# endregion
# =============================================================================


# =============================================================================
# region plot total cloud cover CLCT or low cloud cover CLCL

variables1 = orig_simulation2nd.CLCT[0, 80:920, 80:920].values
variables2 = orig_simulation2nd.CLCL[0, 80:920, 80:920].values
# stats.describe(variables1.flatten())
# np.max(abs(variables1 - variables2))

level = np.arange(0, 1.001, 0.01)
ticks = np.arange(0, 1.001, 0.2)

fig, ax = framework_plot1("1km_lb",)

plt_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + ' ' +
    str(time[timepoint])[11:13] + ':00 UTC')
plt_variables = ax.pcolormesh(
    lon, lat, variables1, cmap=cm.get_cmap('Blues_r', len(level)),
    norm=BoundaryNorm(level, ncolors=len(level), clip=False),
    transform=transform, rasterized=True, zorder=-2,)

cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Total or low type cloud cover [-]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.12 old_sim total_or_low_type_cloud_cover_2010080621.png')


# endregion
# =============================================================================


# =============================================================================
# region plot total cloud cover CLCT or low cloud cover CLCL

variables = orig_simulation.TQC[0, 80:920, 80:920].values
# stats.describe(variables.flatten())

level = np.arange(0, 0.601, 0.01)
ticks = np.arange(0, 0.601, 0.1)

fig, ax = framework_plot1("1km_lb",)

plt_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + ' ' +
    str(time[timepoint])[11:13] + ':00 UTC')
plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('Blues_r', len(level)),
    norm=BoundaryNorm(level, ncolors=len(level), clip=False),
    transform=transform, rasterized=True, zorder=-2,)

cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Vertical integrated cloud water [$kg \; m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_02_time_points_new/6_2_1.13 old_sim vertical_integrated_cloud_water_2010080621.png')


# endregion
# =============================================================================


