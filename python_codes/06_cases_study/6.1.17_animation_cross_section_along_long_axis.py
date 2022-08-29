

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

# region set cross section

upstream_length_c = 1.2
downstream_length_c = 2
startpoint_c = [
    center_madeira[1] + upstream_length_c * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
    center_madeira[0] - upstream_length_c * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
]
endpoint_c = [
    center_madeira[1] - downstream_length_c * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
    center_madeira[0] + downstream_length_c * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
]

# endregion
# =============================================================================


# =============================================================================
# region plot cross section

fig, ax = framework_plot1("1km_lb",)

# ellipse3 = Ellipse(
#     [center_madeira[0] + radius_madeira[1] * 5 * np.cos(
#         np.pi/2 - np.deg2rad(angle_deg_madeira)),
#         center_madeira[1] - radius_madeira[1] * 5 * np.sin(
#         np.pi/2 - np.deg2rad(angle_deg_madeira))],
#     radius_madeira[0] * 2, radius_madeira[1] * 2,
#     angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
# ax.add_patch(ellipse3)

line2, = ax.plot(
    [startpoint_c[1], endpoint_c[1]],
    [startpoint_c[0], endpoint_c[0]],
    lw=0.5, linestyle="-", color='black', zorder=2)
ax.text(startpoint_c[1] - 0.6, startpoint_c[0] + 0.1, '$B_1$')
ax.text(endpoint_c[1] + 0.1, endpoint_c[0], '$B_2$')

fig.savefig('figures/00_test/trial1.png')

# endregion
# =============================================================================


# =============================================================================
# region plot winds on cross section

################################ regrid and combine data
#### regrid wind
strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})
time = strength_direction_3d_20100801_09z.time.values
lon = strength_direction_3d_20100801_09z.lon.values
lat = strength_direction_3d_20100801_09z.lat.values
target_grid = xe.util.grid_2d(
    lon0_b=lon.min(), lon1_b=lon.max(), d_lon=0.01,
    lat0_b=lat.min(), lat1_b=lat.max(), d_lat=0.01)
regridder_wind = xe.Regridder(
    strength_direction_3d_20100801_09z, target_grid, 'bilinear',
    reuse_weights=True)
timepoint = np.where(
    strength_direction_3d_20100801_09z.time ==
    np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
wind_strength = regridder_wind(
    strength_direction_3d_20100801_09z.strength[timepoint])

#### regrid theta -contour
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
regridder_theta = xe.Regridder(
    theta_3d_20100801_09z, target_grid, 'bilinear', reuse_weights=True)
theta = regridder_theta(theta_3d_20100801_09z.theta[timepoint])

#### regrid inversion height -lines
inversion_height = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc')
regridder_hinv = xe.Regridder(
    inversion_height, target_grid, 'bilinear', reuse_weights=True)
hinv = regridder_hinv(inversion_height.inversion_height[timepoint])

#### regrid hsurf
nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
regridder_c = xe.Regridder(
    nc3d_lb_c, target_grid, 'bilinear', reuse_weights=True)
hsurf = regridder_c(nc3d_lb_c.HSURF.squeeze())

################################ create new dataset
ds = xr.merge([wind_strength, theta, hinv, hsurf], compat='override')
dset_cross = ds.metpy.parse_cf()
dset_cross['y'] = dset_cross['lat'].values[:, 0]
dset_cross['x'] = dset_cross['lon'].values[0, :]

################################ create cross section
cross_section_distance1 = distance.distance(startpoint_c, endpoint_c).km
cross1 = cross_section(
    dset_cross, startpoint_c, endpoint_c,
    steps=int(cross_section_distance1/1.1)+1,
).set_coords(('y', 'x'))
x_km1 = np.linspace(0, cross_section_distance1 * 1000, len(cross1.lon))
windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)


################################ plot cross section
# mpl.rc('font', family='Times New Roman', size=10)
fig, ax = plt.subplots(1, 1, figsize=np.array([16, 10]) / 2.54)

#### plot wind
plt_wind = ax.pcolormesh(
    x_km1, cross1.zlev, cross1.strength.values,
    cmap=cm.get_cmap('viridis', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    rasterized=True, zorder=-2,)
cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06,
    shrink=0.6, aspect=30, ticks=ticks, extend='max',)
cbar.ax.set_xlabel('Wind velocity [$m \; s^{-1}$]')

#### plot theta
plt_theta1 = ax.contour(
    x_km1, cross1.zlev, cross1.theta.values,
    colors='black', levels=np.arange(287, 330, 2), linewidths=0.4)
ax.clabel(
    plt_theta1, inline=1, colors='black',
    fmt='%d', levels=np.arange(287, 330, 4), inline_spacing=10, fontsize=8,)

######## plot inversion base height
plt_hinv = ax.plot(
    x_km1, cross1.inversion_height.values, linewidth=1.5, color='red',
    zorder=-2, linestyle='dotted')

plt_topo_bar = ax.fill_between(
    x_km1, cross1.HSURF.values, color='white', zorder=5)
ax.text(
    8000, 4500,
    '$B_1$ - $B_2$  ' + str(time[timepoint])[0:10] + \
    ' ' + str(time[timepoint])[11:13] + ':00 UTC',
    backgroundcolor='white')

ax_legend = ax.legend(
    [plt_theta1.collections[0], plt_hinv[0]],
    ['Potential temperature [K]', 'Inversion base height', ],
    loc='lower center', frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.37),
    handlelength=1, columnspacing=1)
for i in range(len(ax_legend.get_lines())):
    ax_legend.get_lines()[i].set_linewidth(1)

ax.set_yticks(np.arange(0, 5.1, 1) * 1000)
ax.set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax.set_xticks(np.arange(0, 300.1, 50) * 1000)
ax.set_xticklabels(np.arange(0, 300.1, 50, dtype='int'))
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300 km', ],)

ax.set_ylim(0, cross1.zlev.max())
ax.set_ylabel('Height [km]')

fig.subplots_adjust(left=0.07, right=0.96, bottom=0.13, top=0.98)
# fig.savefig('figures/00_test/trial.png', dpi=600)
fig.savefig('figures/06_case_study/6.3.5.0 Vertical cross section along major axis 2010080320.png', dpi=600)



'''
# check regrid

windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)

fig, ax = framework_plot1("1km_lb",)
ax.pcolormesh(
    lon, lat, strength_direction_3d_20100801_09z.strength[timepoint, 1, :, :],
    cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, rasterized=True,
    # zorder=-2,
    )
fig.savefig('figures/00_test/trial2.png')

fig, ax = framework_plot1("1km_lb",)
ax.pcolormesh(
    target_grid.lon, target_grid.lat, wind_strength[1, :, :],
    cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, rasterized=True,
    # zorder=-2,
    )
fig.savefig('figures/00_test/trial1.png')
'''
# endregion
# =============================================================================


# =============================================================================
# region animation winds on cross section

################################ load data
#### regrid wind
strength_direction_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/strength_direction_3d_20100801_09z.nc', chunks={'time': 1})
time = strength_direction_3d_20100801_09z.time.values
lon = strength_direction_3d_20100801_09z.lon.values
lat = strength_direction_3d_20100801_09z.lat.values
target_grid = xe.util.grid_2d(
    lon0_b=lon.min(), lon1_b=lon.max(), d_lon=0.01,
    lat0_b=lat.min(), lat1_b=lat.max(), d_lat=0.01)
regridder_wind = xe.Regridder(
    strength_direction_3d_20100801_09z, target_grid, 'bilinear',
    reuse_weights=True)

#### regrid theta -contour
theta_3d_20100801_09z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/theta_3d_20100801_09z.nc', chunks={'time': 1})
regridder_theta = xe.Regridder(
    theta_3d_20100801_09z, target_grid, 'bilinear', reuse_weights=True)

#### regrid inversion height -lines
inversion_height = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc')
regridder_hinv = xe.Regridder(
    inversion_height, target_grid, 'bilinear', reuse_weights=True)

#### regrid hsurf
nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
regridder_c = xe.Regridder(
    nc3d_lb_c, target_grid, 'bilinear', reuse_weights=True)


################################ plot
istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
ifinal = np.where(time == np.datetime64('2010-08-09T08:00:00.000000000'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-08-03T22:00:00.000000000'))[0][0]

windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 10]) / 2.54, dpi=600)
ims = []

for i in np.arange(istart, ifinal):
    # i = istart
    ################################ create new dataset
    #### extract data
    wind_strength = regridder_wind(
        strength_direction_3d_20100801_09z.strength[i])
    theta = regridder_theta(theta_3d_20100801_09z.theta[i])
    hinv = regridder_hinv(inversion_height.inversion_height[i])
    hsurf = regridder_c(nc3d_lb_c.HSURF.squeeze())
    
    #### merge data
    ds = xr.merge([wind_strength, theta, hinv, hsurf], compat='override')
    dset_cross = ds.metpy.parse_cf()
    dset_cross['y'] = dset_cross['lat'].values[:, 0]
    dset_cross['x'] = dset_cross['lon'].values[0, :]
    
    #### create cross section
    cross_section_distance1 = distance.distance(startpoint_c, endpoint_c).km
    cross1 = cross_section(
        dset_cross, startpoint_c, endpoint_c,
        steps=int(cross_section_distance1/1.1)+1,).set_coords(('y', 'x'))
    x_km1 = np.linspace(0, cross_section_distance1 * 1000, len(cross1.lon))
    
    ################################ plot cross section
    #### plot wind
    plt_wind = ax.pcolormesh(
        x_km1, cross1.zlev, cross1.strength.values,
        cmap=cm.get_cmap('viridis', len(windlevel)),
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        rasterized=True, zorder=-2,)
    
    #### plot theta
    plt_theta = ax.contour(
        x_km1, cross1.zlev, cross1.theta.values,
        colors='black', levels=np.arange(287, 330, 2), linewidths=0.4)
    
    plt_thetalabel = ax.clabel(
        plt_theta, inline=1, colors='black', fmt='%d',
        levels=np.arange(287, 330, 4), inline_spacing=10, fontsize=8,)
    
    #### plot inversion base height
    plt_hinv = ax.plot(
        x_km1, cross1.inversion_height.values, linewidth=1.5, color='red',
        zorder=-2, linestyle='dotted')
    
    #### plot topography
    plt_topo_bar = ax.fill_between(
        x_km1, cross1.HSURF.values, color='white', zorder=5)
    
    #### plot text
    textinfo = ax.text(
        8000, 4500,
        '$B_1$ - $B_2$  ' + str(time[i])[0:10] +
        ' ' + str(time[i])[11:13] + ':00 UTC',
        backgroundcolor='white')
    ims.append(
        plt_hinv + plt_thetalabel + plt_theta.collections +
        [plt_wind, plt_topo_bar, textinfo])
    print(str(i) + '/' + str(ifinal - 1))


cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06,
    shrink=0.6, aspect=30, ticks=ticks, extend='max',)
cbar.ax.set_xlabel('Wind velocity [$m \; s^{-1}$]')

ax_legend = ax.legend(
    [plt_theta.collections[0], plt_hinv[0]],
    ['Potential temperature [K]', 'Inversion base height', ],
    loc='lower center', frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.37),
    handlelength=1, columnspacing=1)
for i in range(len(ax_legend.get_lines())):
    ax_legend.get_lines()[i].set_linewidth(1)

ax.set_yticks(np.arange(0, 5.1, 1) * 1000)
ax.set_yticklabels(np.arange(0, 5.1, 1, dtype='int'))
ax.set_xticks(np.arange(0, 300.1, 50) * 1000)
ax.set_xticklabels(np.arange(0, 300.1, 50, dtype='int'))
ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300 km', ],)

ax.set_ylim(0, cross1.zlev.max())
ax.set_ylabel('Height [km]')

fig.subplots_adjust(left=0.07, right=0.96, bottom=0.13, top=0.98)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    'figures/06_case_study/6.3.5.1 Vertical cross section animation along major axis.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


'''
'''
# endregion
# =============================================================================

