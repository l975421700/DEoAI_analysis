

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

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
# mpl.rcParams['backend'] = "Qt4Agg"
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
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)


# endregion
# =============================================================================


# region file variables check
dir_pgw = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC_PGW/lm_f/'
dir12km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/'
dir1km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/'
dirsatellite = '/store/c2sm/pr04/jvergara/SATELLITE/CMSAF'


#### 1.1 km simulation
nc1h = xr.open_dataset(dir1km + '1h/lffd20051101000000.nc')
nc1h_100m = xr.open_dataset(dir1km + '1h_100m/lffd20051101000000z.nc')
nc1h_second = xr.open_dataset(dir1km + '1h_second/lffd20051101000000.nc')
nc1h_second_c = xr.open_dataset(dir1km + '1h_second/lffd20051101000000c.nc')
nc1h_TOT_PREC = xr.open_dataset(dir1km + '1h_TOT_PREC/lffd20051101010000.nc')
nc1h_TOT_PREC_mm = xr.open_dataset(dir1km + '1h_TOT_PREC_mm/lffd200601.nc')
nc24h = xr.open_dataset(dir1km + '24h/lffd20051101000000.nc')
nc24h_c = xr.open_dataset(dir1km + '24h/lffd20051101000000c.nc')
nc3D_GC = xr.open_dataset(dir1km + '3D_GC/lfsd20051101000000.nc')
nc3D_GC_c = xr.open_dataset(dir1km + '3D_GC/lfsd20051101000000c.nc')
nc3D_Madeira = xr.open_dataset(dir1km + '3D_Madeira/lfsd20100809230000.nc')
nc3D_Madeira_c = xr.open_dataset(dir1km + '3D_Madeira/lfsd20051101000000c.nc')
nc3D_Tenerife = xr.open_dataset(dir1km + '3D_Tenerife/lfsd20051101000000.nc')
nc3D_Tenerife_c = xr.open_dataset(dir1km + '3D_Tenerife/lfsd20051101000000c.nc')
nc3h = xr.open_dataset(dir1km + '3h/lffd20051101000000.nc')
nc3h_CORDEX = xr.open_dataset(dir1km + '3h_CORDEX/lffd20051101000000p.nc')
nc6min_precip = xr.open_dataset(dir1km + '6min_precip/day_lffd20051101.nc')


#### 12 km simulation
nc12km1h = xr.open_dataset(dir12km + '1h/lffd20160101000000.nc')
nc12km_100m = xr.open_dataset(dir12km + '1h_100m/lffd20160101000000z.nc')
nc12km_second = xr.open_dataset(dir12km + '1h_second/lffd20160101000000.nc')
nc12km_second_c = xr.open_dataset(dir12km + '1h_second/lffd20000101000000c.nc')
nc12km_c = xr.open_dataset(dir12km + 'lffd20001001000000c.nc')
(nc12km_second_c.HSURF[0].values == nc12km_c.HSURF[0].values).all()

#### new simulation with 3d output
nc3d_lb = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100801080000.nc')
nc3d_lb_p = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100809230000p.nc')
nc3d_lb_z = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20100810000000z.nc')
nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')


#### external parameters
extpar_file_lm_f = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/MACARONESIA/extpar_0.01_macaron.nc')
# 4000000*840**2/1000**2 * 60 / 2**30 * 24*10 * (3+1)


#### satellite data
nc_TETmm = xr.open_dataset('/store/c2sm/pr04/jvergara/SATELLITE/CMSAF/TETmm201008010000001231000101MA.nc')
nc_TRSmm = xr.open_dataset('/store/c2sm/pr04/jvergara/SATELLITE/CMSAF/TRSmm201008010000001231000101MA.nc')


'''
# check grid
np.sum(nc3d_lb_c.rlat.values == nc1h.rlat[80:920].values)
np.sum(nc3d_lb_c.rlon.values == nc1h.rlon[80:920].values)
np.sum(nc3d_lb_c.lat.values == nc1h.lat[80:920, 80:920].values)
np.sum(nc3d_lb_c.lon.values == nc1h.lon[80:920, 80:920].values)

# check simulation of velocity with Jesus
nc3D_Madeira.rlon[0].values
nc3D_Madeira.rlon[-1].values
np.where(nc3d_lb.rlon == nc3D_Madeira.rlon[0])
nc3d_lb.rlon[520].values
nc3d_lb.rlon[572].values

nc3D_Madeira.rlat[0].values
nc3D_Madeira.rlat[-1].values
np.where(nc3d_lb.rlat == nc3D_Madeira.rlat[-1])
nc3d_lb.rlat[698].values
nc3d_lb.rlat[737].values

np.max(abs(nc3d_lb.U[0, :, 698:(737 + 1), 520:(572 + 1)].values -
           nc3D_Madeira.U[0, :, :, :].values))

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,
)

plt_dem = ax.contourf(
    nc3d_lb.lon, nc3d_lb.lat,
    np.ones(nc3d_lb.lon.shape), colors = 'lightgrey',
    transform=transform,
)

# demlevel = np.arange(0, 20.1, 0.1)
# ticks = np.arange(0, 20.1, 2)
# plt_dem = ax.pcolormesh(
#     nc3d_lb_z.lon, nc3d_lb_z.lat,
#     abs(nc3d_lb_z.U[0, 1, :, :]),
#     norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
#     cmap=cm.get_cmap('terrain', len(demlevel)), rasterized=True,
#     transform=transform,
# )
# cbar = fig.colorbar(
#     plt_dem, orientation="horizontal",  pad=0.1, fraction=0.1,
#     shrink=0.5, aspect=25, ticks=ticks, extend='both')
# cbar.ax.set_xlabel("Topography [m]")

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
plt.savefig('figures/00_test/trial.png', dpi=600)

'''
# endregion


# region check max elevation

nc12km_c = xr.open_dataset('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/lffd20001001000000c.nc')

nc12km_c.HSURF[0, 98:133, 70:117].to_netcdf('scratch/test.nc')

np.max(nc12km_c.HSURF[0, 98:133, 70:117].values)


era5_geop = xr.open_dataset('scratch/obs/era5/adaptor.mars.internal-1661769364.3582895-26853-11-09f36adb-9412-4888-ac5a-5a946ff18a4c.nc')


era5_topograph = mpcalc.geopotential_to_height(
    era5_geop.z.squeeze() * units('meter ** 2 / second ** 2'))
np.max(era5_topograph)

era5_topograph.to_netcdf('scratch/test1.nc')


# endregion
