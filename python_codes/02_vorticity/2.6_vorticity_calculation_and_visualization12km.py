

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
from metpy.calc.thermo import brunt_vaisala_frequency_squared
from haversine import haversine
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
    timing,
    quantiles,
    folder_1km,
    folder_12km,
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
# region calculate 12km vorticity

filelist = np.array(sorted(
    glob.glob(folder_12km + '1h_100m/lffd2010080*.nc')
))


ncfiles = xr.open_mfdataset(
    filelist, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
).metpy.parse_cf()

time = ncfiles.time.values
rlon = ncfiles.rlon.values
rlat = ncfiles.rlat.values
lon = ncfiles.lon.values
lat = ncfiles.lat.values
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

relative_vorticity_12km_1h_100m_20100801_09 = xr.Dataset(
    {"relative_vorticity": (
        ("time", "rlat", "rlon"),
        np.zeros((len(time), len(rlat), len(rlon)))),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "time": time,
        "rlat": rlat,
        "rlon": rlon,
    }
)

u_100m = ncfiles.U.values.squeeze() * units('m/s')
v_100m = ncfiles.V.values.squeeze() * units('m/s')

relative_vorticity_12km_1h_100m_20100801_09.relative_vorticity[:, :, :] = \
    mpcalc.vorticity(u_100m, v_100m, dx[None, :], dy[None, :], dim_order='yx')

relative_vorticity_12km_1h_100m_20100801_09.to_netcdf(
    'scratch/rvorticity/exp_2010/relative_vorticity_12km_1h_100m_20100801_09.nc'
)

# endregion
# =============================================================================


# =============================================================================
# region 12km vorticity animation

# load data
rvorticity_12km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/exp_2010/relative_vorticity_12km_1h_100m_20100801_09.nc'
)
lon = rvorticity_12km_1h_100m.lon.values
lat = rvorticity_12km_1h_100m.lat.values
time = rvorticity_12km_1h_100m.time.values


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


# https://stackoverflow.com/questions/61652069/matplotlib-artistanimation-plot-entire-figure-in-each-step

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 68
ifinal = 200 # 200

for i in np.arange(istart, ifinal):
    rvor = rvorticity_12km_1h_100m.relative_vorticity[
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

scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False)
ax.set_extent(extent1km_lb, crs=transform)
ax.set_rasterization_zorder(-1)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)

ani.save('figures/02_vorticity/2.6.0_Relative_vorticity_12km_in_2010080320_2010080907.mp4')

# endregion
# =============================================================================



