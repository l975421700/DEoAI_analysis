

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

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    hours,
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
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)


# endregion
# =============================================================================


# =============================================================================
# region download files from MESCAN ecmwf

from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()
server.retrieve({
    "class": "ur",
    "dataset": "uerra",
    "date": "2010-08-01/to/2010-08-31",
    "expver": "prod",
    "levtype": "sfc",
    "origin": "lfpw",
    "param": "207/260260",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    "target": "scratch/obs/mescan/mescan_wind_201008.grib",
})


# endregion
# =============================================================================


# =============================================================================
# region check range of ECMWF MESCAN


nc3d_lb = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100809230000.nc')

mescan_wind_201008 = xr.open_dataset(
    'scratch/obs/mescan/mescan_wind_201008.grib', engine='cfgrib')

lon = mescan_wind_201008.longitude.values
lat = mescan_wind_201008.latitude.values
lon[lon > 180] = lon[lon > 180] - 360

lon = lon[50:350, 0:300]
lat = lat[50:350, 0:300]


fig, ax = framework_plot(
    "global", figsize = np.array([8.8, 4.4]) / 2.54, lw=0.1, labelsize = 8
    )

ax.contourf(
    lon, lat, np.ones(lon.shape),
    transform=transform, colors='lightgrey')

ax.contourf(
    nc3d_lb.lon, nc3d_lb.lat, np.ones(nc3d_lb.lon.shape),
    transform=transform, colors='lightblue')

rec_sa = ax.add_patch(Rectangle((-35, 10), 35, 35, ec='red', color='None',
                                lw=0.5))

fig.subplots_adjust(left=0.09, right = 0.96, bottom = 0.08, top = 0.99)
fig.savefig('figures/00_test/trial.png', dpi = 600)



# endregion
# =============================================================================


# =============================================================================
# region calculate velocity of ECMWF MESCAN

mescan_wind_201008 = xr.open_dataset(
    'scratch/obs/mescan/mescan_wind_201008.grib', engine='cfgrib',
    chunks={'time': 5})

lon = mescan_wind_201008.longitude.values
lat = mescan_wind_201008.latitude.values
lon[lon > 180] = lon[lon > 180] - 360
time = mescan_wind_201008.time.values
lon = lon[50:350, 0:300]
lat = lat[50:350, 0:300]

wind_speed = mescan_wind_201008.si10[:, 50:350, 0:300].values
wind_direction = mescan_wind_201008.wdir10[:, 50:350, 0:300].values

u_earth, v_earth = mpcalc.wind_components(
    wind_speed * units('m/s'),
    wind_direction * units.deg)

dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

relative_vorticity = mpcalc.vorticity(
    u_earth, v_earth, dx[None, :], dy[None, :], dim_order='yx')


mescan_wind_201008_rvorticity = xr.Dataset(
    {"relative_vorticity": (("time", "x", "y"), relative_vorticity.magnitude),
     "u_earth": (("time", "x", "y"), u_earth.magnitude),
     "v_earth": (("time", "x", "y"), v_earth.magnitude),
     "lat": (("x", "y"), lat),
     "lon": (("x", "y"), lon),
     },
    coords={
        "time": time,
        "x": np.arange(lon.shape[0]),
        "y": np.arange(lon.shape[1]),
    }
)

mescan_wind_201008_rvorticity.to_netcdf(
    'scratch/obs/mescan/mescan_wind_201008_rvorticity.nc')


# endregion
# =============================================================================


# =============================================================================
# region animate vorticity

mescan_wind_201008_rvorticity = xr.open_dataset(
    'scratch/obs/mescan/mescan_wind_201008_rvorticity.nc')

lon = mescan_wind_201008_rvorticity.lon.values
lat = mescan_wind_201008_rvorticity.lat.values
time = mescan_wind_201008_rvorticity.time.values

# set colormap level and ticks
vorlevel = np.arange(-12, 12.1, 0.1)
ticks = np.arange(-12, 12.1, 3)

# create a color map
top = cm.get_cmap('Blues_r', int(np.floor(len(vorlevel) / 2)))
bottom = cm.get_cmap('Reds', int(np.floor(len(vorlevel) / 2)))
newcolors = np.vstack(
    (top(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2)))),
     [1, 1, 1, 1],
     bottom(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2))))))
newcmp = ListedColormap(newcolors, name='RedsBlues_r')


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 8
ifinal = 35  # 200

for i in np.arange(istart, ifinal):
    rvor = mescan_wind_201008_rvorticity.relative_vorticity[
        i, :, :].values * 10**4
    plt_rvor = ax.pcolormesh(
        lon, lat, rvor, cmap=newcmp, rasterized=True, transform=transform,
        norm=BoundaryNorm(vorlevel, ncolors=newcmp.N, clip=False), zorder=-2,)
    
    rvor_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC')
    
    ims.append([plt_rvor, rvor_time])
    print(str(i) + '/' + str(ifinal - 1))

gl = ax.gridlines(crs=transform, linewidth=0.25, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.add_feature(borders, lw = 0.25); ax.add_feature(coastline, lw = 0.25)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

cbar = fig.colorbar(
    plt_rvor, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Relative vorticity [$10^{-4}\;s^{-1}$]")

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False)
ax.set_extent(extent1km_lb, crs=transform)
ax.set_rasterization_zorder(-1)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)

ani.save('figures/02_vorticity/2.7.0_Relative_vorticity_in_mescan20100803_09.mp4')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate velocity

mescan_wind_201008_rvorticity = xr.open_dataset(
    'scratch/obs/mescan/mescan_wind_201008_rvorticity.nc')

lon = mescan_wind_201008_rvorticity.lon.values
lat = mescan_wind_201008_rvorticity.lat.values
time = mescan_wind_201008_rvorticity.time.values
u_earth = mescan_wind_201008_rvorticity.u_earth.values
v_earth = mescan_wind_201008_rvorticity.v_earth.values


mescan_wind_201008 = xr.open_dataset(
    'scratch/obs/mescan/mescan_wind_201008.grib', engine='cfgrib',
    chunks={'time': 5})
wind_speed = mescan_wind_201008.si10[:, 50:350, 0:300].values
wind_direction = mescan_wind_201008.wdir10[:, 50:350, 0:300].values


windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)

istart = 8
ifinal = 36 # 36

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,
)

ims = []

for i in np.arange(istart, ifinal):
    wind = wind_speed[i, :, :]
    plt_wind = ax.pcolormesh(
        lon, lat, wind, cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        transform=transform, rasterized=True,
    )
    
    iarrow = 6
    windarrow = ax.quiver(
        lon[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
        lat[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
        u_earth[i, int(iarrow/2)::iarrow, int(iarrow/2)::iarrow] /
        (wind[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow]),
        v_earth[i, int(iarrow/2)::iarrow, int(iarrow/2)::iarrow] /
        (wind[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow]),
        scale = 50,
        alpha = 0.5
    )
    
    wind_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC',
        weight = 'bold',
    )
    ims.append([plt_wind, windarrow, wind_time])
    print(str(i) + '/' + str(ifinal - 1))

cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Wind velocity [m/s]")

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )

ax.add_feature(borders, lw = 0.5); ax.add_feature(coastline, lw = 0.5)

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save('figures/03_wind/3.6.0_Wind_strength_direction_in_mescan20100803_09.mp4')


'''
'''


# endregion
# =============================================================================


