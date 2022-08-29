

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


######## plot

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib import patches
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
import rasterio as rio

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
    confidence_ellipse,
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
)


from DEoAI_analysis.module.statistics_calculate import(
    get_statistics,
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
)


# endregion


# region import simulation data ----


nc_1_1h = xr.open_dataset(
    "/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20110811090000.nc")
nc_12_1h = xr.open_dataset(
    "/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h/lffd20000101000000.nc")
nc_1_3d_g = xr.open_dataset(
    "/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_GC/lfsd20051101000000c.nc")
nc_1_3d_m = xr.open_dataset(
    "/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc")
nc_1_3d_t = xr.open_dataset(
    "/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd20051101000000c.nc")

extent1km = [-24.642454, -10.228505, 23.151627, 35.85266]
extent3d_m = [-17.319347, -16.590143, 32.50762, 33.0472]
extent3d_g = [-16.004251, -15.163912, 27.55076, 28.373308]
extent3d_t = [-17.07781, -15.909306, 27.865873, 28.743168]
extent12km = [-30.504213, -4.761099, 17.60372, 40.405067]

# extent1 = [
#     nc_sample1.lon[999, 0].values, nc_sample1.lon[0, 999].values,
#     nc_sample1.lat[0, 0].values, nc_sample1.lat[999, 999].values
# ]
# extent3d_m = [
#     nc_sample2.lon[39, 0].values, nc_sample2.lon[0, 52].values,
#     nc_sample2.lat[0, 0].values, nc_sample2.lat[39, 52].values
# ]
# extent3d_g = [
#     nc_sample4.lon[68, 0].values, nc_sample4.lon[0, 57].values,
#     nc_sample4.lat[0, 0].values, nc_sample4.lat[68, 57].values
# ]
# extent3d_t = [
#     nc_sample5.lon[62, 0].values, nc_sample5.lon[0, 90].values,
#     nc_sample5.lat[0, 0].values, nc_sample5.lat[62, 90].values
# ]
# extent12 = [
#     nc_sample3.lon[164, 0].values, nc_sample3.lon[0, 164].values,
#     nc_sample3.lat[0, 0].values, nc_sample3.lat[164, 164].values
# ]

# endregion
# =============================================================================


# =============================================================================
# region plot global and study area ----
ticklabel = ticks_labels(-180, 180, -90, 90, 60, 30)
extent = [-180, 180, -90, 90]
transform = ctp.crs.PlateCarree()

fig, ax = plt.subplots(
    1, 1, figsize = np.array([8.8, 4.4]) / 2.54,
    subplot_kw={'projection': transform})
ax.set_extent(extent, crs = transform)
ax.set_xticks(ticklabel[0])
ax.set_xticklabels(ticklabel[1])
ax.set_yticks(ticklabel[2])
ax.set_yticklabels(ticklabel[3])

gl = ax.gridlines(crs = transform, linewidth = 0.1,
                  color = 'gray', alpha = 0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel[0])
gl.ylocator = mticker.FixedLocator(ticklabel[2])

coastline = ctp.feature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw = 0.1)
ax.add_feature(coastline)
borders = ctp.feature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
    facecolor='none', lw = 0.1)
ax.add_feature(borders)

# scale_bar(ax, bars = 2, length = 1000, location = (0.1, 0.05),
#           barheight = 60, linewidth = 0.2, col = 'black')

rec_sa = ax.add_patch(Rectangle((-35, 10), 35, 35, ec = 'red', color = 'None',
                               lw = 0.5))

# fig.legend([rec_sa], ['study area'], loc = 'lower center',
#            frameon = False, ncol = 2, bbox_to_anchor = (0.5, -0.05))
fig.subplots_adjust(left=0.12, right = 0.94, bottom = 0.1, top = 0.99)
fig.savefig('figures/01_study_area/1.1.0 global and study area.png', dpi=600)


'''
fig, ax = framework_plot(
    "global", figsize = np.array([8.8, 4.4]) / 2.54, lw=0.1, labelsize = 8
    )

rec_sa = ax.add_patch(Rectangle((-35, 10), 35, 35, ec='red', color='None',
                                lw=0.5))

fig.subplots_adjust(left=0.09, right = 0.96, bottom = 0.08, top = 0.99)
fig.savefig('figures/00_test/trial.png', dpi = 600)
'''
# endregion
# =============================================================================


# =============================================================================
# region plot study area ----
ticklabel = ticks_labels(-30, 0, 10, 40, 10, 10)
extent = [-35, 0, 10, 45]
transform = ctp.crs.PlateCarree()

fig, ax = plt.subplots(
    1, 1, figsize = np.array([8.8, 9.6]) / 2.54,
    subplot_kw={'projection': transform})
ax.set_extent(extent, crs = transform)
ax.set_xticks(ticklabel[0])
ax.set_xticklabels(ticklabel[1])
ax.set_yticks(ticklabel[2])
ax.set_yticklabels(ticklabel[3])

gl = ax.gridlines(crs = transform, linewidth = 0.2,
                  color = 'gray', alpha = 0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel[0])
gl.ylocator = mticker.FixedLocator(ticklabel[2])

coastline = ctp.feature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw = 0.2)
ax.add_feature(coastline)
borders = ctp.feature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
    facecolor='none', lw = 0.2)
ax.add_feature(borders)

scale_bar(ax, bars = 2, length = 1000, location = (0.05, 0.025),
          barheight = 60, linewidth = 0.2, col = 'black')

# model_lb = ax.contourf(
#     nc3d_lb_c.lon, nc3d_lb_c.lat, np.ones(nc3d_lb_c.lon.shape),
#     transform=transform, colors='gainsboro', alpha=0.5, zorder=-3)

plt_12km = ax.contourf(
    nc_12_1h.lon, nc_12_1h.lat, np.ones(nc_12_1h.lon.shape),
    transform=transform, colors='gainsboro', alpha=0.5, zorder=-3)
plt_1km = ax.contourf(
    nc_1_1h.lon, nc_1_1h.lat, np.ones(nc_1_1h.lon.shape),
    transform=transform, colors='gainsboro', alpha=1, zorder=-3)
h1,_ = plt_12km.legend_elements()
h2,_ = plt_1km.legend_elements()
rec_m = ax.add_patch(Rectangle((-18, 32), 2.5, 1.5, ec = 'red', color = 'None',
                               lw = 0.5))
rec_c = ax.add_patch(Rectangle((-18.5, 27.5), 5.1, 2, ec = 'blue',
                               color = 'None', lw = 0.5))

fig.legend([h1[0], h2[0], rec_m, rec_c],
           ['12 km simulation', '1.1 km simulation', 'Madeira Islands',
            'Canary Islands'], loc = 'lower center', frameon = False, ncol = 2)
fig.subplots_adjust(left=0.13, right = 0.96, bottom = 0.2, top = 0.99)
fig.savefig('figures/01_study_area/1.1 study area.png', dpi=600)


# endregion
# =============================================================================


# =============================================================================
# region plot Madeira ----

with rio.open(
    'data_source/topograph/eudem_dem_5deg_n30w020/eudem_dem_5deg_n30w020.tif',
    masked = True) as madeira_dem:
    madeira_dem_data = madeira_dem.read(1)
    madeira_dem_bounds = madeira_dem.bounds
    madeira_dem_data[madeira_dem_data == 0] = np.nan
    # np.sum(np.isnan(madeira_dem_data) )
    madeira_dem_data = np.ma.array(madeira_dem_data)
    madeira_dem_data[np.isnan(madeira_dem_data)] = np.ma.masked


nc3D_Madeira_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc')
lon = np.ma.array(nc3D_Madeira_c.lon.values)
lat = np.ma.array(nc3D_Madeira_c.lat.values)
hsurf_m = np.ma.array(nc3D_Madeira_c.HHL[0, -1, :, :].values)
lon[hsurf_m == 0] = np.ma.masked
lat[hsurf_m == 0] = np.ma.masked
hsurf_m[hsurf_m == 0] = np.ma.masked

lon_lat = np.transpose(np.vstack((lon[~lon.mask].data, lat[~lat.mask].data)))
clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
clf.fit(lon_lat)

mean = clf.means_[0, :]
covar = clf.covariances_[0, :, :]
v, w = linalg.eigh(covar)
v = 2. * np.sqrt(2.) * np.sqrt(v)
u = w[0] / linalg.norm(w[0])

angle_rad = np.arctan(u[1] / u[0])
angle_deg = np.rad2deg(angle_rad)

# calculate four vertices
ne_vertices = [mean[0] + v[0] * 0.7 * np.cos(angle_rad),
               mean[1] + v[0] * 0.7 * np.sin(angle_rad)]
sw_vertices = [mean[0] - v[0] * 0.7 * np.cos(angle_rad),
               mean[1] - v[0] * 0.7 * np.sin(angle_rad)]
nw_vertices = [mean[0] - v[1] * 0.7 * np.cos(np.pi/2 - angle_rad),
               mean[1] + v[1] * 0.7 * np.sin(np.pi/2 - angle_rad)]
se_vertices = [mean[0] + v[1] * 0.7 * np.cos(np.pi/2 - angle_rad),
               mean[1] - v[1] * 0.7 * np.sin(np.pi/2 - angle_rad)]

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 9]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=False)

#### plot ellipse
ellipse1 = Ellipse(mean, v[0] * 1.4, v[1] * 1.4, angle_deg,
                   edgecolor='red', facecolor='none', lw=0.75)
plt_madeira_e = ax.add_patch(ellipse1)
# ax.scatter(
#     np.mean(lon[~lon.mask].data), np.mean(lat[~lat.mask].data),
#     c='red', s=3, zorder=2)
# ax.scatter(ne_vertices[0], ne_vertices[1], c='red', s=3, zorder=2)
# ax.scatter(sw_vertices[0], sw_vertices[1], c='red', s=3, zorder=2)
# ax.scatter(nw_vertices[0], nw_vertices[1], c='red', s=3, zorder=2)
# ax.scatter(se_vertices[0], se_vertices[1], c='red', s=3, zorder=2)

#### plot a smaller ellipse
se1_center = [
    mean[0] + 2 * v[0] * 0.7 * np.cos(angle_rad),
    mean[1] + 2 * v[0] * 0.7 * np.sin(angle_rad)]
s_ellipse1 = Ellipse(se1_center, v[0] * 1.4 / 3, v[1] * 1.4 / 3, angle_deg,
                   edgecolor='blue', facecolor='none', lw=0.75)
plt_se1 = ax.add_patch(s_ellipse1)
plt_se1_center = ax.scatter(
    se1_center[0], se1_center[1], c='blue', s=2.5, zorder=2)

# se2_center = [
#     mean[0] - 2*v[0]*0.7 * np.cos(angle_rad) * 0.9 - \
#         v[1] * 0.7/3 * np.sin(angle_rad),
#     mean[1] - 2*v[0]*0.7 * np.sin(angle_rad) * 0.9 + \
#         v[1] * 0.7/3 * np.cos(angle_rad)]
# s_ellipse2 = Ellipse(se2_center, v[0] * 1.4 / 3, v[1] * 1.4 / 3, angle_deg,
#                    edgecolor='c', facecolor='none', lw=0.75)
# plt_se2 = ax.add_patch(s_ellipse2)
# plt_se2_center = ax.scatter(
#     se2_center[0], se2_center[1], c='c', s=2.5, zorder=2)



#### plot arrows
p1 = patches.FancyArrowPatch(
    ne_vertices, sw_vertices, arrowstyle='<->', mutation_scale=5,
    lw=0.5, color='red', shrinkA=0, shrinkB=0, zorder=2)
ax.add_patch(p1)
p2 = patches.FancyArrowPatch(
    nw_vertices, se_vertices, arrowstyle='<->', mutation_scale=5,
    lw=0.5, color='red', shrinkA=0, shrinkB=0, zorder=2)
ax.add_patch(p2)

#### plot dashed lines and an arc
line1, = ax.plot(
    [ne_vertices[0], ne_vertices[0] + v[0] * 0.7 * np.cos(angle_rad)],
    [ne_vertices[1], ne_vertices[1] + v[0] * 0.7 * np.sin(angle_rad)],
    lw=0.5, linestyle="--", color='red', zorder=2)
line2, = ax.plot(
    [ne_vertices[0], ne_vertices[0] + v[0] * 0.7],
    [ne_vertices[1], ne_vertices[1]],
    lw=0.5, linestyle="--", color='red', zorder=2)
e1 = patches.Arc(ne_vertices, width=v[0] * 0.5, height=v[0] * 0.5,
                 angle=0, theta1=0, theta2=angle_deg,
                 linewidth=0.5, color='red',
                 fill=False, zorder=2)
ax.add_patch(e1)

#### plot 3d simulation and points
# 3d simulation domain
plt_3dm = ax.contourf(
    nc_1_3d_m.lon, nc_1_3d_m.lat, np.ones(nc_1_3d_m.lon.shape),
    transform=transform, colors='gainsboro', alpha = 0.5)
h1,_ = plt_3dm.legend_elements()
# Pico Ruivo altitude, 1862 m
plt_pico = ax.scatter(
    -16.943192382588784, 32.75966092711838, marker='^', s=4, c='black',
    zorder = 10)
# Paul da Serra altitude, 1500 m
plt_paul = ax.scatter(
    -17.050000039841176, 32.734198597157956, marker='v', s=4, c='black',
    zorder=10)
# Funchal sounding altitude, 58 m
plt_funchal = ax.scatter(
    -16.9000, 32.6333, marker='^', s=4, c='red', zorder=10)
ax_legend = ax.legend(
    [h1[0], plt_madeira_e, plt_pico, plt_funchal, plt_se1, plt_paul, ],
    ['3D model outputs', 'Ellipse $e_1$ fitted to Madeira', 'Pico Ruivo, 1862 m a.s.l.',
    'Funchal radiosonde from IGRA2', 'Small ellipse $se$ with center $O$', 'Paul da Serra, 1500 m a.s.l.',],
    loc = 'lower center', frameon = False, ncol = 2, fontsize = 8,
    bbox_to_anchor = (0.43, -0.66), handlelength = 1,
    columnspacing = 1)

#### plot topography
lon1, lat1 = np.meshgrid(
    np.linspace(
        madeira_dem_bounds.left, madeira_dem_bounds.right,
        int(madeira_dem_data.shape[1])),
    np.linspace(
        madeira_dem_bounds.bottom, madeira_dem_bounds.top,
        int(madeira_dem_data.shape[1])),
    sparse=True)
demlevel = np.arange(0, 1800.1, 10)
ticks = np.arange(0, 1800.1, 300)
plt_dem = ax.pcolormesh(
    lon1[0, 9000:14400], lat1[11700:8100:-1, 0],
    madeira_dem_data[6300:9900, 9000:14400],
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(demlevel)), rasterized=True,
    transform=transform,)
# terrain
cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.1, fraction=0.12,
    shrink=0.8, aspect=25, ticks=ticks, extend='max')
cbar.ax.set_xlabel("Topography [m]")

#### plot text
ax.text(-17.3, 32.9, 'Madeira', fontsize=8)
ax.text(-16.5, 32.95, 'Porto Santo', fontsize=8)
ax.text(-16.5, 32.55, 'Desertas', fontsize=8)
ax.text(-17, 32.6,
        str(int(np.round(
            distance.distance([ne_vertices[1], ne_vertices[0]],
                              [sw_vertices[1], sw_vertices[0]]).km, 0))) + \
        ' km', fontsize=8)
ax.text(-16.7, 32.65,
        str(int(np.round(
            distance.distance([nw_vertices[1], nw_vertices[0]],
                              [se_vertices[1], se_vertices[0]]).km, 0))) + \
        ' km', fontsize=8)
ax.text(
    # ne_vertices[0] + v[0] * 0.25, ne_vertices[1] + v[0] * 0.25,
    -16.93, 32.84,
    '79°',
    fontsize=8)

########
scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.97, bottom=0.21, top=0.99)
# fig.savefig('figures/00_test/trial.png', dpi=300)
# fig.savefig('figures/01_study_area/1.2 Madeira 3d.png', dpi=600)
fig.savefig('figures/01_study_area/1.2 Madeira.png', dpi=600)

'''
ax.text(-17.05, 32.87, '$M_{1}$')
ax.text(-17, 32.59, '$M_{2}$')
ax.text(-17.32, 32.75, '$M_{3}$')
ax.text(-16.71, 32.68, '$M_{4}$')
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Canary islands ----

with rio.open(
    'data_source/topograph/eudem_dem_5deg_n25w020/eudem_dem_5deg_n25w020.tif',
    masked = True) as tenerife_dem:
    tenerife_dem_data = tenerife_dem.read(1)
    tenerife_dem_bounds = tenerife_dem.bounds
    tenerife_dem_data[tenerife_dem_data == 0] = np.nan
    tenerife_dem_data = np.ma.array(tenerife_dem_data)
    tenerife_dem_data[np.isnan(tenerife_dem_data)] = np.ma.masked

with rio.open(
    'data_source/topograph/eudem_dem_5deg_n25w015/eudem_dem_5deg_n25w015.tif',
    masked = True) as canary_dem:
    canary_dem_data = canary_dem.read(1)
    canary_dem_bounds = canary_dem.bounds
    canary_dem_data[canary_dem_data == 0] = np.nan
    canary_dem_data = np.ma.array(canary_dem_data)
    canary_dem_data[np.isnan(canary_dem_data)] = np.ma.masked

transform = ctp.crs.PlateCarree()
ticklabel = ticks_labels(-18, -14, 27.5, 29.5, 1, 0.5)
extent = [-18.2, -13.2, 27.5, 29.5]
# [-17.5, -16, 32.25, 33.25]
# [9000, 14400, 9900, 6300]

fig, ax = plt.subplots(
    1, 1, figsize = np.array([15.6, 9]) / 2.54,
    subplot_kw={'projection': transform})
ax.set_extent(extent, crs = transform)
ax.set_xticks(ticklabel[0])
ax.set_xticklabels(ticklabel[1])
ax.set_yticks(ticklabel[2])
ax.set_yticklabels(ticklabel[3])

gl = ax.gridlines(crs = transform, linewidth = 0.5,
                  color = 'gray', alpha = 0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel[0])
gl.ylocator = mticker.FixedLocator(ticklabel[2])
scale_bar(ax, bars = 2, length = 100, location = (0.7, 0.05),
          barheight = 6, linewidth = 0.2, col = 'black')

plt_3dg = ax.contourf(
    nc_1_3d_g.lon, nc_1_3d_g.lat, np.ones(nc_1_3d_g.lon.shape),
    transform=transform, colors='gainsboro', alpha=0.5)
plt_3dt = ax.contourf(
    nc_1_3d_t.lon, nc_1_3d_t.lat, np.ones(nc_1_3d_t.lon.shape),
    transform=transform, colors='gainsboro', alpha=0.5)
h1,_ = plt_3dg.legend_elements()
# ax.legend([h1[0]], ['3D Model Simulation'], loc = 'upper left', frameon = False)

plt_tenerife = ax.scatter(
    -16.3822, 28.3183, marker='^', s=4, c='red', zorder=10)
ax_legend = ax.legend(
    [h1[0], plt_tenerife, ],
    ['3D model outputs', 'Tenerife radiosonde from IGRA2', ],
    loc='lower center', frameon=False, ncol=2, fontsize=10,
    bbox_to_anchor=(0.5, -0.55), handlelength=1,
    columnspacing=1)

lon, lat = np.meshgrid(
    np.linspace(
        tenerife_dem_bounds.left, tenerife_dem_bounds.right,
        int(tenerife_dem_data.shape[1])),
    np.linspace(
        tenerife_dem_bounds.bottom, tenerife_dem_bounds.top,
        int(tenerife_dem_data.shape[1])),
    sparse=True)
lon1, lat1 = np.meshgrid(
    np.linspace(
        canary_dem_bounds.left, canary_dem_bounds.right,
        int(canary_dem_data.shape[1])),
    np.linspace(
        canary_dem_bounds.bottom, canary_dem_bounds.top,
        int(canary_dem_data.shape[1])),
    sparse=True)

demlevel = np.arange(0, 3000.1, 10)
ticks = np.arange(0, 3000.1, 500)

plt_dem = ax.pcolormesh(
    lon[0, 3600:18000], lat[16200:9000:-1, 0],
    tenerife_dem_data[1800:9000, 3600:18000],
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(demlevel)), rasterized=True,
    transform=transform,)
plt_dem1 = ax.pcolormesh(
    lon1[0, 0:7200], lat1[16200:9000:-1, 0],
    canary_dem_data[1800:9000, 0:7200],
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(demlevel)), rasterized=True,
    transform=transform,)

cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.1, fraction=0.1,
    shrink=0.5, aspect=25, ticks=ticks, extend='max')
cbar.ax.set_xlabel("Topography [m]")

ax.text(-17, 28.6, 'Tenerife')
ax.text(-16.6, 27.8, 'Gran Canaria')

ax.text(-17.7, 28.8, 'La Palma')
ax.text(-17.9, 27.6, 'El Hierro')
ax.text(-18, 28.1, 'La Gomera')
ax.text(-15, 28.3, 'Fuerteventura')
ax.text(-14.2, 29.2, 'Lanzarote')


fig.subplots_adjust(left=0.09, right = 0.99, bottom = 0.15, top = 0.99)

fig.savefig('figures/01_study_area/1.3 Canary 3d.png', dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region plot study area , fitted ellipse and cross section

nc3D_Madeira_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20051101000000c.nc')
nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
hsurf = np.ma.array(nc3d_lb_c.HSURF.squeeze().values)
hsurf[hsurf == 0] = np.ma.masked

demlevel = np.arange(0, 2500.1, 10)
ticks = np.arange(0, 2500.1, 500)

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54, country_boundaries=False,
    gridlines=True,)

################################ plot model DEM

plt_dem = ax.pcolormesh(
    nc3d_lb_c.lon, nc3d_lb_c.lat, hsurf,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(demlevel)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.1, fraction=0.1,
    shrink=1, aspect=25, ticks=ticks, extend='max')
cbar.ax.set_xlabel("Topography [m] in CRS1")

################################ plot ellipse
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

# ellipse4 = Ellipse(
#     [center_madeira[0] - radius_madeira[0] * 3 * np.cos(
#         np.deg2rad(angle_deg_madeira)),
#         center_madeira[1] - radius_madeira[0] * 3 * np.sin(
#         np.deg2rad(angle_deg_madeira))],
#     radius_madeira[0] * 2, radius_madeira[1] * 2,
#     angle_deg_madeira, edgecolor='red', facecolor='none', lw=0.5)
# ax.add_patch(ellipse4)

ax.text(center_madeira[0] - 0.9, center_madeira[1] - 0.3, '$e_1$')
ax.text(center_madeira[0] - 0.6, center_madeira[1] + 0.6, '$e_2$')
ax.text(center_madeira[0] + 1.2, center_madeira[1], '$e_3$')
# ax.text(center_madeira[0] + 0.1, center_madeira[1] - 0.7, '$e_4$')

################################ plot cross section

# upstream_length = 1.2
# downstream_length = 3
# startpoint = [
#     center_madeira[1] + upstream_length * np.sin(
#         np.deg2rad(angle_deg_madeira + 0)),
#     center_madeira[0] + upstream_length * np.cos(
#         np.deg2rad(angle_deg_madeira + 0)),
#     ]
# endpoint = [
#     center_madeira[1] - downstream_length * np.sin(
#         np.deg2rad(angle_deg_madeira + 0)),
#     center_madeira[0] - downstream_length * np.cos(
#         np.deg2rad(angle_deg_madeira + 0)),
#     ]
# line1, = ax.plot(
#     [startpoint[1], endpoint[1]],
#     [startpoint[0], endpoint[0]],
#     lw=0.5, linestyle="-", color='black', zorder=2)

# upstream_length_c = 1.2
# downstream_length_c = 2
# startpoint_c = [
#     center_madeira[1] + upstream_length_c * np.sin(
#         np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
#     center_madeira[0] - upstream_length_c * np.cos(
#         np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
#     ]
# endpoint_c = [
#     center_madeira[1] - downstream_length_c * np.sin(
#         np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
#     center_madeira[0] + downstream_length_c * np.cos(
#         np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
#     ]
# line2, = ax.plot(
#     [startpoint_c[1], endpoint_c[1]],
#     [startpoint_c[0], endpoint_c[0]],
#     lw=0.5, linestyle="-", color='black', zorder=2)

# ax.text(startpoint[1], startpoint[0] + 0.1, '$A_1$')
# ax.text(endpoint[1] + 0.1, endpoint[0], '$A_2$')
# ax.text(startpoint_c[1] - 0.6, startpoint_c[0] + 0.1, '$B_1$')
# ax.text(endpoint_c[1] + 0.1, endpoint_c[0], '$B_2$')

################################ plot model area
model_lb = ax.contourf(
    nc3d_lb_c.lon, nc3d_lb_c.lat, np.ones(nc3d_lb_c.lon.shape),
    transform=transform, colors='gainsboro', alpha=0.25, zorder=-3)

################################ plot Madeira mask

# ax.scatter(nc3d_lb_c.lon[300, 0], nc3d_lb_c.lat[300, 0], s = 2)
# ax.scatter(nc3d_lb_c.lon[390, 320], nc3d_lb_c.lat[390, 320], s = 2)
# ax.scatter(nc3d_lb_c.lon[260, 580], nc3d_lb_c.lat[260, 580], s = 2)
# ax.scatter(nc3d_lb_c.lon[380, 839], nc3d_lb_c.lat[380, 839], s=2)
# ax.scatter(nc3d_lb_c.lon[839, 839], nc3d_lb_c.lat[839, 839], s=2)
# ax.scatter(nc3d_lb_c.lon[839, 0], nc3d_lb_c.lat[839, 0], s=2)
nc1h_second_c = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20051101000000c.nc')
lon_a = nc1h_second_c.lon.values
lat_a = nc1h_second_c.lat.values

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

ax.contour(
    lon_a, lat_a, mask_data, colors='black', levels=np.array([0.5]),
    linewidths=1, linestyles='dashed')
ax.contour(
    lon_a, lat_a, mask_data1, colors='r', levels=np.array([0.5]),
    linewidths=0.5, linestyles='solid')

ax.add_feature(borders, lw=0.2)
ax.add_feature(coastline, lw=0.2)
scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
# plt.savefig('figures/01_study_area/1.4 study area_details.png', dpi=600)
# plt.savefig('figures/01_study_area/1.5 study area_details_subset.png', dpi=600)
plt.savefig('figures/01_study_area/1.4.0 study area_details_with_mask.png',
            dpi=600)
# plt.savefig('figures/01_study_area/1.4.1 study area_details_only_mask.png',
#             dpi=600)


'''
import pylab as plt
import numpy as np
from matplotlib.path import Path
width, height=2000, 2000

polygon=[(0.1*width, 0.1*height), (0.15*width, 0.7*height), (0.8*width, 0.75*height), (0.72*width, 0.15*height)]
poly_path=Path(polygon)

x, y = np.mgrid[:height, :width]
coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

mask = poly_path.contains_points(coors)
plt.imshow(mask.reshape(height, width))
plt.show()



# Distance A1 to A2
distance.distance(startpoint, endpoint).km # 463 km
distance.distance(startpoint_c, endpoint_c).km  # 302 km

madeira_100 = ax.pcolormesh(
    nc3d_lb_c.lon[698:(698 + 60), 520:(520 + 60)],
    nc3d_lb_c.lat[698:(698 + 60), 520:(520 + 60)],
    np.ones(nc3d_lb_c.lon[698:(698 + 60), 520:(520 + 60)].shape),
    # lon[698:798, 520:620], lat[698:798, 520:620],
    # np.ones(lon[698:798, 520:620].shape),
    transform=transform, color='lightgrey', zorder=-3)

madeira_org = ax.contourf(
    nc3D_Madeira_c.lon, nc3D_Madeira_c.lat,
    np.ones(nc3D_Madeira_c.lon.shape),
    transform=transform, colors='grey')

######## based on lat-lon pairs
# polygon = [
#     (-20.939148, 26.952396), (-18.053928, 28.94064),
#     (-14.848157, 28.563076), (-12.517615, 30.467636),
#     (-14.128447, 34.85296), (-23.401758, 31.897812),
# ]
# poly_path = Path(polygon)
# coors = np.hstack((nc3d_lb_c.lon.values.reshape(-1, 1),
#                    nc3d_lb_c.lat.values.reshape(-1, 1)))
# mask = poly_path.contains_points(coors).reshape(840, 840)

mask_data = np.ma.array(np.ones(nc3d_lb_c.lon.shape))
mask_data.mask = (mask == False)
mask_area = ax.contourf(
    nc3d_lb_c.lon, nc3d_lb_c.lat, mask_data,
    transform=transform, colors='gainsboro', alpha=0.75, zorder=-3)

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Madeira ----


nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
hsurf = np.ma.array(nc3d_lb_c.HSURF.squeeze().values)
hsurf[hsurf == 0] = np.ma.masked

fig, ax = framework_plot(
    "madeira", figsize=np.array([8.8, 9]) / 2.54, lw=0.25, labelsize=10,
    country_boundaries=False)

demlevel = np.arange(0, 1800.1, 10)
ticks = np.arange(0, 1800.1, 300)
plt_dem = ax.pcolormesh(
    nc3d_lb_c.lon, nc3d_lb_c.lat, hsurf,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(demlevel)), rasterized=True,
    transform=transform,)
cbar = fig.colorbar(
    plt_dem, orientation="horizontal",  pad=0.1, fraction=0.12,
    shrink=0.8, aspect=25, ticks=ticks, extend='max')
cbar.ax.set_xlabel("Topography [m] in CRS1")

#### plot text
ax.text(-17.3, 32.9, 'Madeira', fontsize=8)
ax.text(-16.5, 32.95, 'Porto Santo', fontsize=8)
ax.text(-16.5, 32.55, 'Desertas', fontsize=8)

########
scale_bar(ax, bars=2, length=20, location=(0.05, 0.05),
          barheight=1, linewidth=0.2, col='black')
fig.subplots_adjust(left=0.155, right=0.97, bottom=0.21, top=0.99)
fig.savefig('figures/01_study_area/1.2.1 Madeira_CRS1_topography.png', dpi=600)

'''
'''
# endregion
# =============================================================================

