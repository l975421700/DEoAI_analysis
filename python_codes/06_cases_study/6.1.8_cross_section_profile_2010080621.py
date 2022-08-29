

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
import tables as tb
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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['backend'] = 'Qt4Agg'  #
# mpl.get_backend()
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.rcParams["pcolor.shading"] = 'auto'

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
    framework_plot1,
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
    rvor_level,
    rvor_ticks,
    rvor_colors,
    rvor_cmp,
)


from DEoAI_analysis.module.statistics_calculate import(
    get_statistics,
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
)

from DEoAI_analysis.module.meteo_calc import(
    lifting_condensation_level
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

ellipse1 = Ellipse(
    center_madeira,
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse1)

ellipse2 = Ellipse(
    [center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
        center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira))],
    radius_madeira[0] * 2, radius_madeira[1] * 2,
    angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
ax.add_patch(ellipse2)

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

# add cross section line2
lineangle3 = angle_deg_madeira
startpoint3 = [
    center_madeira[1] + upstream_length * np.sin(np.deg2rad(lineangle3 + 0)),
    center_madeira[0] + upstream_length * np.cos(np.deg2rad(lineangle3 + 0)),
]
endpoint3 = [
    center_madeira[1] - downstream_length * np.sin(np.deg2rad(lineangle3 + 0)),
    center_madeira[0] - downstream_length * np.cos(np.deg2rad(lineangle3 + 0)),
]

# endregion
# =============================================================================


ihours = 141


# =============================================================================
# region 20100806 21 UTC cross section at z level

################################ regrid theta -contour
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
lon = theta_3d_20100801_09z.lon.values
lat = theta_3d_20100801_09z.lat.values
target_grid = xe.util.grid_2d(
    lon0_b=lon.min(), lon1_b=lon.max(), d_lon=0.01,
    lat0_b=lat.min(), lat1_b=lat.max(), d_lat=0.01)
regridder_theta = xe.Regridder(
    theta_3d_20100801_09z, target_grid, 'bilinear', reuse_weights=True)
theta = regridder_theta(theta_3d_20100801_09z.theta[ihours])
# np.max(theta.values); np.max(theta_3d_20100801_09z.theta[ihours].values)

################################ regrid relative vorticity
rvorticity_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/rvorticity_3d_20100801_09z.nc')
regridder_rvor = xe.Regridder(
    rvorticity_3d_20100801_09z, target_grid, 'bilinear', reuse_weights=True)
rvor = regridder_rvor(rvorticity_3d_20100801_09z.relative_vorticity[ihours])

################################ regrid hsurf -black
nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
regridder_c = xe.Regridder(
    nc3d_lb_c, target_grid, 'bilinear', reuse_weights=True)
hsurf = regridder_c(nc3d_lb_c.HSURF.squeeze())

################################ regrid wind -contourf
strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})
regridder_wind = xe.Regridder(
    strength_direction_3d_20100801_09z, target_grid, 'bilinear',
    reuse_weights=True)
wind_strength = regridder_wind(
    strength_direction_3d_20100801_09z.strength[ihours])

################################ regrid inversion height -lines
inversion_height = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc')
regridder_hinv = xe.Regridder(
    inversion_height, target_grid, 'bilinear', reuse_weights=True)
hinv = regridder_hinv(inversion_height.inversion_height[ihours])

################################ regrid PBL height -lines
################################ lifting condensation level
ncfile = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20100806210000.nc')
regridder_hpbl = xe.Regridder(
    ncfile, target_grid, 'bilinear', reuse_weights=True)
hpbl = regridder_hpbl(ncfile.HPBL.squeeze())

# hpbl1 = regridder_hpbl(ncfile.HPBL.squeeze().values)
lcl_ps = ncfile.PS[0, 80:920, 80:920].values
lcl_relhum_2m = ncfile.RELHUM_2M[0, 80:920, 80:920].values / 100
lcl_tem_2m = ncfile.T_2M[0, 80:920, 80:920].values
lcl = np.zeros_like(lcl_ps)
for i in range(lcl.shape[0]):
    for j in range(lcl.shape[1]):
        lcl[i, j] = lifting_condensation_level(
            lcl_ps[i, j], lcl_tem_2m[i, j], lcl_relhum_2m[i, j])
    # print(str(i) + '/' + str(lcl.shape[0]))
# stats.describe(lcl.flatten())
lcl_regrid = regridder_c(lcl)
lcl_xr = hsurf.copy()
lcl_xr.name = 'LCL'
lcl_xr.values = lcl_regrid

'''
# check
i = 300
j =400
lifting_condensation_level(lcl_ps[i, j], lcl_tem_2m[i, j], lcl_relhum_2m[i, j])
lcl[i, j]
lcl_david(lcl_ps[i, j], lcl_tem_2m[i, j], lcl_relhum_2m[i, j])
'''

################################ regrid relative humidity -contour
nc3d_lb_z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100806210000z.nc')
regridder_3d_zsim = xe.Regridder(
    nc3d_lb_z, target_grid, 'bilinear', reuse_weights=True)
# qc = regridder_3d_zsim(nc3d_lb_z.QC.squeeze())
relhum = regridder_3d_zsim(nc3d_lb_z.RELHUM.squeeze())

################################ regrid cloud water liquid content -contour
qc_zlev_height = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/qc_zlev_height.nc')
regridder_qc = xe.Regridder(
    qc_zlev_height, target_grid, 'bilinear', reuse_weights=True)
qc = regridder_qc(qc_zlev_height.QC[ihours])


# create new dataset
ds = xr.merge(
    [theta, hsurf, wind_strength, hinv,
     rvor, hpbl, lcl_xr, qc, relhum],
    compat='override')
dset_cross = ds.metpy.parse_cf()
dset_cross['y'] = dset_cross['lat'].values[:, 0]
dset_cross['x'] = dset_cross['lon'].values[0, :]


################################ create cross section 1
cross_section_distance1 = distance.distance(startpoint1, endpoint1).km
cross1 = cross_section(
    dset_cross,
    startpoint1,
    endpoint1,
    steps=int(cross_section_distance1/1.1)+1,
).set_coords(('y', 'x'))
x_km1 = np.linspace(0, cross_section_distance1 * 1000, len(cross1.lon))
windlevel = np.arange(0, 15.1, 0.1)
ticks = np.arange(0, 15.1, 3)
# cross.theta.values


################################ create cross section 2
cross_section_distance2 = distance.distance(startpoint2, endpoint2).km
cross2 = cross_section(
    dset_cross,
    startpoint2,
    endpoint2,
    steps=int(cross_section_distance2/1.1)+1,
).set_coords(('y', 'x'))
x_km2 = np.linspace(0, cross_section_distance2 * 1000, len(cross2.lon))
windlevel = np.arange(0, 15.1, 0.1)
ticks = np.arange(0, 15.1, 3)

cross1.load()
cross2.load()


# =============================================================================
################################ plot first set of variables

fig = plt.figure(figsize=np.array([16, 15]) / 2.54)
gs = fig.add_gridspec(2, 1, hspace=0.05, wspace=0.1)
ax = gs.subplots(sharex = True)

################ plot C1 - C2
######## plot wind
plt_wind1 = ax[0].pcolormesh(
    x_km1, cross1.zlev, cross1.strength,
    cmap=cm.get_cmap('viridis', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    rasterized=True, zorder=-2,)

######## plot theta
plt_theta1 = ax[0].contour(
    x_km1, cross1.zlev, cross1.theta,
    colors='black', levels=np.arange(287, 319, 2), linewidths=0.4)
ax[0].clabel(plt_theta1, inline=1, colors='black',
             fmt='%d', levels=np.arange(287, 319, 4), inline_spacing=10,
             fontsize=8,)

######## plot inversion base height
ax[0].plot(
    x_km1, cross1.inversion_height.values, linewidth=1.5, color='red',
    zorder = -2, linestyle='dotted')

######## plot PBL height
# ax[0].plot(
#     x_km1, cross1.HPBL.values, linewidth=1, color='blue', linestyle='--',
#     )

######## plot topography
ax[0].fill_between(
    x_km1, cross1.HSURF.values, color='black',)

######## plot axis
ax[0].text(8000, 4500, '$C_1$ - $C_2$', backgroundcolor='white')
ax[0].set_yticks(np.arange(0, 5.1, 1) * 1000)
ax[0].set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax[0].set_ylim(0, cross1.zlev.max())
ax[0].set_ylabel('Height [km]')
ax[0].set_xticks(np.arange(0, 450.1, 50) * 1000)
# ax[0].set_xticklabels(np.arange(0, 450.1, 50, dtype='int'))

################ plot C3 - C4
######## plot wind
plt_wind2 = ax[1].pcolormesh(
    x_km2, cross2.zlev, cross2.strength,
    cmap=cm.get_cmap('viridis', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=True),
    rasterized=True, zorder=-2,)

######## plot theta
plt_theta2 = ax[1].contour(
    x_km2, cross2.zlev, cross2.theta,
    colors='black', levels=np.arange(287, 319, 2), linewidths=0.4)

ax[1].clabel(plt_theta2, inline=1, colors='black',
          fmt='%d', levels=np.arange(287, 319, 4), inline_spacing=10,
          fontsize=8,)


######## plot inversion base height
plt_hinv, = ax[1].plot(
    x_km2, cross2.inversion_height.values, linewidth=1.5, color='red',
    zorder=-2, linestyle='dotted')

######## plot PBL height
# ax[1].plot(
#     x_km2, cross2.HPBL.values, linewidth=1, color='blue', linestyle='--',
#     )

######## plot topography
ax[1].fill_between(
    x_km2, cross2.HSURF.values, color='black',
    )

######## plot axis
ax[1].text(8000, 4500, '$C_3$ - $C_4$', backgroundcolor='white')
ax[1].set_yticks(np.arange(0, 5.1, 1) * 1000)
ax[1].set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax[1].set_ylim(0, cross2.zlev.max())
ax[1].set_ylabel('Height [km]')
ax[1].set_xticks(np.arange(0, 450.1, 50) * 1000)

################ plot color bar
cbar = fig.colorbar(
    plt_wind2, ax=ax, pad=0.05, fraction=0.15, anchor = (1.2, 0.5),
    shrink=0.6, aspect=30, ticks=ticks, extend='max',
    )
cbar.ax.set_ylabel('Wind velocity [$m \; s^{-1}$]')
ax[1].set_xticklabels(
    ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450 km', ],
    )

# ax[1].set_xlabel(
#     'Wind velocity [m/s] and potential temperature [K] at 2010-08-06 21:00 UTC')
ax_legend = ax[1].legend([plt_theta2.collections[0], plt_hinv],
                      ['Potential temperature [K]',
                       'Inversion base height', ],
                      loc='lower center', frameon=False, ncol=2,
                      bbox_to_anchor=(0.5, -0.25), handlelength=1,
                      columnspacing=1)
for i in range(len(ax_legend.get_lines())):
    ax_legend.get_lines()[i].set_linewidth(1)

fig.subplots_adjust(left=0.07, right=0.89, bottom=0.12, top=0.98)
fig.savefig(
    'figures/06_case_study/6.3.4.0 Vertical cross section of first set of variables in 06.08.2010 21.00 UTC.png', dpi=600)


# =============================================================================
################################ plot Second set of variables
rvor_level = np.arange(-12, 12.1, 0.12)
rvor_ticks = np.arange(-12, 12.1, 3)
rvor_cmp = ListedColormap(rvor_colors[20:-20, :], name='RedsBlues_r')
hinv_linec = 'red'
lcl_linec = 'black'
relhum_linec = 'blue'
cloud_linec = 'black'
ymax = 2500

fig = plt.figure(figsize=np.array([16, 15]) / 2.54)
gs = fig.add_gridspec(2, 1, hspace=0.05, wspace=0.1)
ax = gs.subplots(sharex = True)

################ plot A5 - A6
######## plot relative vorticity
plt_rvor1 = ax[0].pcolormesh(
    x_km1, cross1.zlev, cross1.relative_vorticity * 10**4, cmap=rvor_cmp,
    rasterized=True, zorder=-2,
    norm=BoundaryNorm(rvor_level, ncolors=len(rvor_level), clip=False),)

######## plot relative humidity
plt_relhum1 = ax[0].contour(
    x_km1, cross1.zlev, cross1.RELHUM, colors=relhum_linec, linewidths=1,
    levels=[95], zorder=-1,)
# def stipple(pCube, thresh=0.05, central_long=0):
#     """
#     Stipple points using plt.scatter for values below thresh in pCube.
#     If you have used a central_longitude in the projection, other than 0,
#     this must be specified with the central_long keyword
#     """
#     xOrg = pCube.coord('longitude').points
#     yOrg = pCube.coord('latitude').points
#     nlon = len(xOrg)
#     nlat = len(yOrg)
#     xData = np.reshape( np.tile(xOrg, nlat), pCube.shape )
#     yData = np.reshape( np.repeat(yOrg, nlon), pCube.shape )
#     sigPoints = pCube.data < thresh
#     xPoints = xData[sigPoints] - central_long
#     yPoints = yData[sigPoints]
#     plt.scatter(xPoints,yPoints,s=1, c='k', marker='.', alpha=0.5)


# def stipple(x1, x2, data, threshold, lower_bound = True):
#     dim1, dim2 = data.shape
    
#     xdata = np.reshape(np.tile(x2, dim1), data.shape)
#     ydata = np.reshape(np.repeat(x1, dim2), data.shape)
#     if lower_bound:
#         sigPoints = data > threshold
#     else:
#         sigPoints = data < threshold
    
#     xpoints = xdata[sigPoints]
#     ypoints = ydata[sigPoints]
    
#     return (xpoints, ypoints)
#     # plt.scatter(xpoints, ypoints, s=1, c='k', marker='.', alpha=0.5)


# xpoints, ypoints = stipple(
#     x1=cross1.zlev.values, x2=x_km1, data=cross1.RELHUM.values, threshold=80)
# ax[0].scatter(xpoints, ypoints, s=1, c='k', marker='.', alpha=0.5)
'''
x1=cross1.zlev.values
x2=x_km1
data=cross1.RELHUM.values
threshold = 95
lower_bound = True
'''
# plt_relhum1 = ax[0].contourf(
#     x_km1, cross1.zlev, cross1.RELHUM,
#     colors='none', linewidths=1,
#     levels=[90, 100],
#     zorder=-1,
#     hatches = '////'
# )

######## plot inversion base height
ax[0].plot(
    x_km1, cross1.inversion_height.values, linewidth=1.5, color=hinv_linec,
    zorder=-2, linestyle='dotted')

######## plot PBL height
# ax[0].plot(
#     x_km1, cross1.HPBL.values, linewidth=1, color=hinv_linec, linestyle='--',
#     zorder=-2,
#     )

######## plot LCL
ax[0].plot(
    x_km1[np.where(cross1.HSURF.values == 0)[0]],
    cross1.LCL.values[np.where(cross1.HSURF.values == 0)[0]],
    linewidth=1, color=lcl_linec,
    zorder=-2,)

######## plot cloud
ax[0].contour(
    x_km1, cross1.height, cross1.QC,
    levels=[0.0001],
    colors=cloud_linec, linewidths=1.5, linestyles='dotted')


######## plot topography
plt_topo_bar1 = ax[0].fill_between(
    x_km1, cross1.HSURF.values, color='black', zorder = 2,)
######## plot axis
ax[0].text(8000, ymax*0.9, '$C_1$ - $C_2$', backgroundcolor='white')
ax[0].set_yticks(np.arange(0, ymax + 1, 500))
# ax[0].set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax[0].set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
ax[0].set_ylim(0, ymax)
ax[0].set_ylabel('Height [km]')
ax[0].set_xticks(np.arange(0, 450.1, 50) * 1000)

################ plot A3 - A4
######## plot relative vorticity
plt_rvor2 = ax[1].pcolormesh(
    x_km2, cross2.zlev, cross2.relative_vorticity * 10**4, cmap=rvor_cmp,
    rasterized=True, zorder=-2,
    norm=BoundaryNorm(rvor_level, ncolors=len(rvor_level), clip=False),)

######## plot relative humidity
plt_relhum2 = ax[1].contour(
    x_km2, cross2.zlev, cross2.RELHUM,
    colors=relhum_linec, linewidths=1,
    levels=[95],
    zorder=-1,)
# ax[1].clabel(plt_relhum2, inline=1, colors='black',
#              fmt='%d', levels=np.arange(0, 101, 10), inline_spacing=10,
#              fontsize=8,
#              )

######## plot inversion base height
plt_hinv = ax[1].plot(
    x_km2, cross2.inversion_height.values, linewidth=1.5, color=hinv_linec,
    zorder=-2, linestyle='dotted')

######## plot PBL height
# ax[1].plot(
#     x_km2, cross2.HPBL.values, linewidth=1, color=hinv_linec, linestyle='--',
#     zorder=-2,
#     )

######## plot LCL
plt_lcl = ax[1].plot(
    x_km2[np.where(cross2.HSURF.values == 0)[0]],
    cross2.LCL.values[np.where(cross2.HSURF.values == 0)[0]],
    linewidth=1, color=lcl_linec,
    zorder=-2,)

# plot cloud
plt_cloud = ax[1].contour(
    x_km2, cross2.height, cross2.QC,
    levels=[0.0001],
    colors=cloud_linec, linewidths=1.5, linestyles='dotted')

######## plot topography
plt_topo_bar2 = ax[1].fill_between(
    x_km2, cross2.HSURF.values, color='black', zorder=2,)

######## plot axis
ax[1].text(8000, ymax*0.9, '$C_3$ - $C_4$', backgroundcolor='white')
ax[1].set_yticks(np.arange(0, ymax + 1, 500))
# ax[1].set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax[1].set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
ax[1].set_ylim(0, ymax)
ax[1].set_ylabel('Height [km]')
ax[1].set_xticks(np.arange(0, 450.1, 50) * 1000)
ax[1].set_xticklabels(
    ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450 km', ],)

################ plot color bar
cbar = fig.colorbar(
    plt_rvor2, ax=ax, pad=0.05, fraction=0.15, anchor=(1.1, 0.5),
    shrink=0.6, aspect=30, ticks=rvor_ticks, extend='both',
    )
cbar.ax.set_ylabel('Relative vorticity [$10^{-4} \; s^{-1}$]')

ax_legend = ax[1].legend(
    [plt_hinv[0], plt_lcl[0], plt_relhum2.collections[0],
     plt_cloud.collections[0]],
    ['Inversion base height', 'Lifting condensation level',
     'Relative humidity [95%]',
     'Specific cloud liquid water content [$10^{-4} \; kg\;kg^{-1}$]',
     ],
                         loc='lower center', frameon=False, ncol=2,
                         bbox_to_anchor=(0.5, -0.3), handlelength=1,
                         columnspacing=1)
for i in range(len(ax_legend.get_lines())):
    ax_legend.get_lines()[i].set_linewidth(1)

# ax[1].set_xlabel(
#     'Relative vorticity [$10^{-4}\;s^{-1}$] and relative humidity [%] at 2010-08-06 21:00 UTC')
fig.subplots_adjust(left=0.08, right=0.89, bottom=0.12, top=0.98)
fig.savefig(
    'figures/06_case_study/6.3.4.1 Vertical cross section of second set of variables in 06.08.2010 21.00 UTC.png', dpi=600)


'''
################################ plot cross section 1
# mpl.rc('font', family='Times New Roman', size=10)
fig, ax = plt.subplots(1, 1, figsize=np.array([16, 10]) / 2.54)

# plot wind
plt_wind = ax.pcolormesh(
    x_km1, cross1.zlev, cross1.strength,
    cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    rasterized=True, zorder=-2,
)
cbar = fig.colorbar(
    plt_wind, ax=ax, pad=0.01, fraction=0.06,
    shrink=0.8, aspect=30, ticks=ticks, extend='neither',
)

plt_theta = ax.contour(
    x_km1, cross1.zlev, cross1.theta,
    colors='black', levels=np.arange(287, 319, 2), linewidths=0.3
)
ax.clabel(plt_theta, inline=1, colors='black',
          fmt='%d', levels=np.arange(287, 319, 4), inline_spacing=10,
          fontsize=8,
          )

plt_topo_bar = ax.fill_between(
    x_km1, cross1.HSURF.values, color='black',
    )

ax.set_yticks(np.arange(0, 5.1, 1) * 1000)
ax.set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax.set_xticks(np.arange(0, 450.1, 50) * 1000)
ax.set_xticklabels(np.arange(0, 450.1, 50, dtype='int'))
ax.set_ylim(0, cross1.zlev.max())

ax.set_xlabel(
    'Wind velocity [m/s] and potential temperature [K] in 06.08.2010 21:00 UTC from $A_3$ to $A_4$ [km]')
ax.set_ylabel('Height [km]')

fig.subplots_adjust(left=0.07, right=0.999, bottom=0.11, top=0.98)
fig.savefig(
    'figures/06_case_study/6.3.2 Wind velocity and potential temperature in 06.08.2010 21.00 UTC from A3 to A4.png',
    dpi=600)
'''

'''
################################ plot cross section 2
# mpl.rc('font', family='Times New Roman', size=10)
fig, ax = plt.subplots(1, 1, figsize=np.array([16, 10]) / 2.54)

# plot wind
plt_wind = ax.pcolormesh(
    x_km2, cross2.zlev, cross2.strength,
    cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    rasterized=True, zorder=-2,
)
cbar = fig.colorbar(
    plt_wind, ax=ax, pad=0.01, fraction=0.06,
    shrink=0.8, aspect=30, ticks=ticks, extend='neither',
)

plt_theta = ax.contour(
    x_km2, cross2.zlev, cross2.theta,
    colors='black', levels=np.arange(287, 319, 2), linewidths=0.3
)
ax.clabel(plt_theta, inline=1, colors='black',
          fmt='%d', levels=np.arange(287, 319, 4), inline_spacing=10,
          fontsize=8,
          )

plt_topo_bar = ax.fill_between(
    x_km2, cross2.HSURF.values, color='black',
)

ax.set_yticks(np.arange(0, 5.1, 1) * 1000)
ax.set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax.set_xticks(np.arange(0, 450, 50) * 1000)
ax.set_xticklabels(np.arange(0, 450, 50, dtype='int'))
ax.set_ylim(0, cross2.zlev.max())

ax.set_xlabel(
    'Wind velocity [m/s] and potential temperature [K] in 06.08.2010 21:00 UTC from $A_5$ to $A_6$ [km]')
ax.set_ylabel('Height [km]')

fig.subplots_adjust(left=0.07, right=0.999, bottom=0.11, top=0.98)
fig.savefig(
    'figures/06_case_study/6.3.3 Wind velocity and potential temperature in 06.08.2010 21.00 UTC from A5 to A6.png',
    dpi=600)
'''

# endregion
# =============================================================================

