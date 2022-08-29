

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


iyear = str('10')
imonth = str('08')
iday = str('07')
ihour = str('12')


# old simulation
# =============================================================================
# =============================================================================
# region load data

# rvor
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + iyear + imonth + '.nc')
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values

# wind
wind_100m = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + iyear + imonth + '.nc')

# original simulation
orig_simulation = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' + \
    iyear + imonth + iday + ihour + '0000.nc')
orig_simulation2nd = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20' +
    iyear + imonth + iday + ihour + '0000.nc')

pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

# old simulation constant file
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
timepoint = np.where(
    rvorticity_1km_1h_100m.time == np.datetime64(
        '20' + iyear + '-' + imonth + '-' + iday + 'T' + ihour))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[timepoint, 80:920, 80:920].values
wind_v = wind_100m.v_earth[timepoint, 80:920, 80:920].values


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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.0 rvor_' + '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.1 2mtem_' +
    '20' + iyear + imonth + iday + ihour + '.png')

# endregion
# =============================================================================


# =============================================================================
# region plot surface pressure

pres = orig_simulation.PS[0, 80:920, 80:920].values / 100
# stats.describe(pres.flatten()) # [681, 1020]
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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.2 surface_pressure_' +
    '20' + iyear + imonth + iday + ihour + '.png')


# endregion
# =============================================================================


# =============================================================================
# region plot theta

# stats.describe(theta.flatten()) # [280, 305]
theta_mid = 293
theta_min = theta_mid - 3
theta_max = theta_mid + 3

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.3 surface_theta_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.11 100_m_winds_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.4 averaged_surface_sensible_heat_flux_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.5 averaged_surface_latent_heat_flux_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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

cbar_level = np.arange(500, 850.01, 1)
cbar_ticks = np.arange(500, 850.01, 50)


fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('viridis', len(cbar_level)),
    rasterized=True, transform=transform,
    norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
cbar = fig.colorbar(
    plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Averaged surface net downward SW radiation [$W\;m^{-2}$]")

ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_06_time_points_extend/6_6.0.6 averaged_surface_net_downward_shortwave_radiation_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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

cbar_level = np.arange(750, 1100.01, 1)
cbar_ticks = np.arange(750, 1100.01, 50)

fig, ax = framework_plot1("1km_lb",)

timeplot = ax.text(
    -23, 34, str(time[timepoint])[0:10] +
    ' ' + str(time[timepoint])[11:13] + ':00 UTC')

plt_variables = ax.pcolormesh(
    lon, lat, variables, cmap=cm.get_cmap('viridis', len(cbar_level)),
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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.7 averaged_TOA_net_downward_shortwave_radiation_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.8 averaged_TOA_outgoing_longwave_radiation_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.9 averaged_surface_net_downward_longwave_radiation_' +
    '20' + iyear + imonth + iday + ihour + '.png')

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

# variables = orig_simulation2nd.LWU_S[0, 80:920, 80:920].values
# # stats.describe(variables.flatten())

# cbar_mid = 430
# cbar_min = cbar_mid - 15
# cbar_max = cbar_mid + 15
# cbar_level = np.arange(cbar_min, cbar_max + 0.01, 0.1)
# cbar_ticks = np.arange(cbar_min, cbar_max + 0.01, 5)

# cbar_top = cm.get_cmap('Blues_r', int(np.floor(len(cbar_level) / 2)))
# cbar_bottom = cm.get_cmap('Reds', int(np.floor(len(cbar_level) / 2)))
# cbar_colors = np.vstack(
#     (cbar_top(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2)))),
#      [1, 1, 1, 1],
#      cbar_bottom(np.linspace(0, 1, int(np.floor(len(cbar_level) / 2))))))
# cbar_cmp = ListedColormap(cbar_colors, name='RedsBlues_r')

# fig, ax = framework_plot1("1km_lb",)

# timeplot = ax.text(
#     -23, 34, str(time[timepoint])[0:10] + \
#     ' ' + str(time[timepoint])[11:13] + ':00 UTC')

# plt_variables = ax.pcolormesh(
#     lon, lat, variables, cmap=cbar_cmp,
#     rasterized=True, transform=transform,
#     norm=BoundaryNorm(cbar_level, ncolors=len(cbar_level), clip=False), )
# cbar = fig.colorbar(
#     plt_variables, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
#     shrink=0.9, aspect=25, ticks=cbar_ticks, extend='both',
#     anchor=(0.5, 1), panchor=(0.5, 0))
# cbar.ax.set_xlabel(
#     "Upward lw radiation at the surface [$W\;m^{-2}$]")

# ax.contour(lon, lat, is_vortex_trs,
#            colors='lime', levels=np.array([-0.5, 0.5]),
#            linewidths=0.5, linestyles='solid')
# ax.contourf(lon, lat, model_topo_mask,
#             colors='white', levels=np.array([0.5, 1.5]))

# fig.savefig(
#     'figures/06_case_study/06_02_time_points_new/6_2_1.10 old_sim upward_lw_radiation_at_the_surface_2010080621.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.12 total_or_low_type_cloud_cover_' +
    '20' + iyear + imonth + iday + ihour + '.png')


# endregion
# =============================================================================


# =============================================================================
# region plot Vertical integrated cloud water TQC

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
    'figures/06_case_study/06_06_time_points_extend/6_6.0.13 vertical_integrated_cloud_water_' +
    '20' + iyear + imonth + iday + ihour + '.png')


# endregion
# =============================================================================


iyear = str('10')
imonth = str('08')
iday = str('05')
ihour = str('11')
# =============================================================================
# region load data

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

# rvor
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + iyear + imonth + '.nc')
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values

# wind
wind_100m = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + iyear + imonth + '.nc')

# original simulation
orig_simulation = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' + \
    iyear + imonth + iday + ihour + '0000.nc')
orig_simulation2nd = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20' +
    iyear + imonth + iday + ihour + '0000.nc')

pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

# old simulation constant file
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
timepoint = np.where(
    rvorticity_1km_1h_100m.time == np.datetime64(
        '20' + iyear + '-' + imonth + '-' + iday + 'T' + ihour))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[timepoint, 80:920, 80:920].values
wind_v = wind_100m.v_earth[timepoint, 80:920, 80:920].values


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
    'figures/06_case_study/06_06_time_points_extend/6_6.1.0 rvor_' + '20' + iyear + imonth + iday + ihour + '.png')

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
    'figures/06_case_study/06_06_time_points_extend/6_6.1.11 100_m_winds_' +
    '20' + iyear + imonth + iday + ihour + '.png')

# endregion
# =============================================================================


# =============================================================================
# region plot total cloud cover CLCT or low cloud cover CLCL

variables1 = orig_simulation2nd.CLCT[0, 80:920, 80:920].values
variables2 = orig_simulation2nd.CLCL[0, 80:920, 80:920].values
# stats.describe(variables1.flatten())
# np.mean(abs(variables1 - variables2))

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
cbar.ax.set_xlabel("Total cloud cover [-]")

# ax.contour(lon, lat, is_vortex_trs,
#            colors='lime', levels=np.array([-0.5, 0.5]),
#            linewidths=0.5, linestyles='solid'
#            )
ax.contourf(lon, lat, model_topo_mask,
            colors='black', levels=np.array([0.5, 1.5]))
ax.contour(
    lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')
fig.savefig(
    'figures/06_case_study/06_06_time_points_extend/6_6.1.12 total_or_low_type_cloud_cover_' +
    '20' + iyear + imonth + iday + ihour + '.png')


# endregion
# =============================================================================


# =============================================================================
# region plot Vertical integrated cloud water TQC

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

# ax.contour(lon, lat, is_vortex_trs,
#            colors='lime', levels=np.array([-0.5, 0.5]),
#            linewidths=0.5, linestyles='solid'
#            )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_06_time_points_extend/6_6.1.13 vertical_integrated_cloud_water_' +
    '20' + iyear + imonth + iday + ihour + '.png')


# endregion
# =============================================================================
