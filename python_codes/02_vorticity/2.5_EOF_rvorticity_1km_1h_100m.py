


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


# region calculation

folder_rvorticity_1km_1h_100m = \
    'scratch/relative_vorticity_1km_1h_100m'

filelist_rvorticity_1km_1h_100m = \
    sorted(glob.glob(folder_rvorticity_1km_1h_100m + '/*100m2010*'))


rvorticity_1km_1h_100m = xr.open_mfdataset(
    filelist_rvorticity_1km_1h_100m, concat_dim="time",
    data_vars='minimal', coords='minimal', parallel=True
)

time = rvorticity_1km_1h_100m.time


from eofs.standard import Eof

# ddd = np.array(rvorticity_1km_1h_100m.relative_vorticity[0:744, 80:920, 80:920])
# weights_array = np.cos(np.deg2rad(np.array(
#     rvorticity_1km_1h_100m.lat[80:920, 80:920])))
# rvorticity_1km_1h_100m_solver = Eof(ddd, weights=weights_array)

# with open('scratch/rvorticity_1km_1h_100m_solver.pickle', 'wb') as f:
#     pickle.dump(rvorticity_1km_1h_100m_solver, f)

with open('scratch/rvorticity_1km_1h_100m_solver.pickle', 'rb') as f:
    rvorticity_1km_1h_100m_solver = pickle.load(f)

pcs = rvorticity_1km_1h_100m_solver.pcs(npcs=10)
eofs = rvorticity_1km_1h_100m_solver.eofs(neofs=10)
eigenvalues = rvorticity_1km_1h_100m_solver.eigenvalues(neigs=10)
variance_fractions = rvorticity_1km_1h_100m_solver.varianceFraction(neigs=10)

# endregion


# region visulization

rvorticity_1km_1h_100m_statistics = xr.open_dataset(
    'scratch/rvorticity/exp_2010/relative_vorticity_1km_1h_100m2010_statistics.nc')
lon = rvorticity_1km_1h_100m_statistics.lon[:, :].values
lat = rvorticity_1km_1h_100m_statistics.lat[:, :].values

with open('scratch/rvorticity/exp_2010/rvorticity_1km_1h_100m_solver.pickle', 'rb') as f:
    rvorticity_1km_1h_100m_solver = pickle.load(f)


# set colormap level and ticks
vorlevel = np.arange(-12, 12.1, 0.1)
ticks = np.arange(-12, 12.1, 3)

# create a color map
top = cm.get_cmap('Blues_r', int(np.floor(len(vorlevel) / 2)))
bottom = cm.get_cmap('Reds', int(np.floor(len(vorlevel) / 2)))
newcolors = np.vstack((top(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2)))),
                       [1, 1, 1, 1],
                       bottom(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2))))))
newcmp = ListedColormap(newcolors, name='RedsBlues_r')

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
    subplot_kw={'projection': transform})

plt_rvor = ax.pcolormesh(
    lon, lat,
    rvorticity_1km_1h_100m_solver.eofs()[2, :, :] * 10**3,
    cmap=newcmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(vorlevel, ncolors=len(vorlevel), clip=False),
    )

cbar = fig.colorbar(
    plt_rvor, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='both')
cbar.ax.set_xlabel(
    "EOF2 of relative vorticity in 201001 [$10^{-3}s^{-1}$]")

# EOF1(0.0146); EOF2(0.00966)

ax.add_feature(borders); ax.add_feature(coastline)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False)

gl = ax.gridlines(crs=transform, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.set_extent(extent1km_lb, crs=transform)

fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig(
    'figures/02_vorticity/2.1.1 relative vorticity in 201001_EOF2.png', dpi=1200)
plt.close('all')


'''

'''

# endregion











