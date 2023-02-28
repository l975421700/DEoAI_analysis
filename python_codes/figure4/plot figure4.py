

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
# mpl.get_backend()
plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

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
import Ngl,Nio
import netCDF4 as nc

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
# =============================================================================


# =============================================================================
# region import data

ncols=3
nlon=17
nlat=1
nrows=nlon*nlat

stat_validation = {}
stat_validation['am_pre'] = {}
stat_validation['am_pre']['Stations'] = Ngl.asciiread(
    'DEoAI_analysis/python_codes/figure4/(a)/stations/pr_Stat_2006-2015_yearly_accumulated_MAD.out',
    (nrows,ncols),
    "float")
stat_validation['am_pre']['CPS12'] = Ngl.asciiread(
    'DEoAI_analysis/python_codes/figure4/(a)/12km/pr_MAD_Madeira_2006-2015_yearly_accumulated_MAD.out',
    (nrows,ncols),
    "float")
stat_validation['am_pre']['CRS1'] = Ngl.asciiread(
    'DEoAI_analysis/python_codes/figure4/(a)/1km/pr_MAD_Madeira_2006-2015_yearly_accumulated_MAD.out',
    (nrows,ncols),
    "float")

stat_validation['pre_bias'] = {}
stat_validation['pre_bias']['CPS12'] = Ngl.asciiread(
    'DEoAI_analysis/python_codes/figure4/(b)/12km/pr_MAD_Madeira_v_Stat_2006-2015_yearly_accumulated_relative_error_MAD.out',
    (nrows,ncols),
    "float")
stat_validation['pre_bias']['CRS1'] = Ngl.asciiread(
    'DEoAI_analysis/python_codes/figure4/(b)/1km/pr_MAD_Madeira_v_Stat_2006-2015_yearly_accumulated_relative_error_MAD.out',
    (nrows,ncols),
    "float")

shpfilename = 'DEoAI_analysis/python_codes/figure4/shapefile/gadm36_PRT_0.shp'
shape_feature = ShapelyFeature(
    Reader(shpfilename).geometries(),
    ccrs.PlateCarree(), facecolor='none',edgecolor='k', lw=0.5)

# endregion
# =============================================================================


# =============================================================================
# region plot data

output_png = 'figures/11_figure4/11.0_validation against station observations.png'

cbar_label1 = 'Annual precipitation [mm]'
cbar_label2 = 'Precipitation biases [%]'

pltlevel = np.arange(0, 2000 + 1e-4, 200)
pltticks = np.arange(0, 2000 + 1e-4, 400)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1)

pltlevel2 = np.arange(-100, 100 + 1e-4, 20)
pltticks2 = np.arange(-100, 100 + 1e-4, 20)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1)
extent = [-17.3, -16.65, 32.6, 32.9]
ticklabel = ticks_labels(-17.3, -16.7, 32.6, 32.9, 0.2, 0.1)
# coastline = ctp.feature.NaturalEarthFeature(
#     'physical', 'coastline', '10m', edgecolor='black',
#     facecolor='none', lw=0.5)

nrow = 2
ncol = 3
fm_bottom = 2 / (2.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 3*nrow + 2]) / 2.54,
    sharex=True, sharey=True,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

panel_labels=['(a)', '(b)', '(c)', '(d)', '(e)']
ipanel=0

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 1) | (jcol != 0)):
            axs[irow, jcol].set_extent(extent, crs = ccrs.PlateCarree())
            # axs[irow, jcol].set_xticks(ticklabel[0])
            # axs[irow, jcol].set_xticklabels(ticklabel[1])
            # axs[irow, jcol].set_yticks(ticklabel[2])
            # axs[irow, jcol].set_yticklabels(ticklabel[3])
            # axs[irow, jcol].tick_params(length=2)
            # axs[irow, jcol].add_feature(coastline, zorder=2)
            axs[irow, jcol].add_feature(shape_feature, zorder=2)
            gl = axs[irow, jcol].gridlines(
                crs=transform, linewidth=0.25, zorder=2,
                color='gray', alpha=0.5, linestyle='--')
            gl.xlocator = mticker.FixedLocator(ticklabel[0])
            gl.ylocator = mticker.FixedLocator(ticklabel[2])

            plt.text(
                0, 1.08, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes, weight='bold',
                ha='left', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

axs[0, 0].scatter(
    stat_validation['am_pre']['Stations'][:, 0],
    stat_validation['am_pre']['Stations'][:, 1],
    c=stat_validation['am_pre']['Stations'][:, 2],
    s=8, cmap=pltcmp, transform=ccrs.PlateCarree(),
    norm=pltnorm, zorder=2)

axs[0, 1].scatter(
    stat_validation['am_pre']['CPS12'][:, 0],
    stat_validation['am_pre']['CPS12'][:, 1],
    c=stat_validation['am_pre']['CPS12'][:, 2],
    s=8, cmap=pltcmp, transform=ccrs.PlateCarree(),
    norm=pltnorm, zorder=2)

axs[0, 2].scatter(
    stat_validation['am_pre']['CRS1'][:, 0],
    stat_validation['am_pre']['CRS1'][:, 1],
    c=stat_validation['am_pre']['CRS1'][:, 2],
    s=8, cmap=pltcmp, transform=ccrs.PlateCarree(),
    norm=pltnorm, zorder=2)

axs[1, 1].scatter(
    stat_validation['pre_bias']['CPS12'][:, 0],
    stat_validation['pre_bias']['CPS12'][:, 1],
    c=stat_validation['pre_bias']['CPS12'][:, 2],
    s=8, cmap=pltcmp2, transform=ccrs.PlateCarree(),
    norm=pltnorm2, zorder=2)

axs[1, 2].scatter(
    stat_validation['pre_bias']['CRS1'][:, 0],
    stat_validation['pre_bias']['CRS1'][:, 1],
    c=stat_validation['pre_bias']['CRS1'][:, 2],
    s=8, cmap=pltcmp2, transform=ccrs.PlateCarree(),
    norm=pltnorm2, zorder=2)


# axs[0, 0].set_title('Stations')
plt.text(
    0.5, 1.08, 'Stations', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.08, 'CPS12', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.08, 'CRS1', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.08, '(CPS12 - Stations) / Stations', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.08, '(CRS1 - Stations) / Stations', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(1.1,-3.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.96)
fig.savefig(output_png)

# endregion
# =============================================================================

