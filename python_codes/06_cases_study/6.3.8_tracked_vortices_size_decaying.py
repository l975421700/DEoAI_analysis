

# =============================================================================
# region import packages


# basic library
from matplotlib.path import Path
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

from DEoAI_analysis.module.vortex_namelist import (
    correctly_reidentified,
    track_p1, track_p2, track_p3, track_p4,
    track_n1, track_n2, track_n3, track_n4,
)

# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
from scipy import ndimage
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py
from scipy.ndimage.filters import median_filter, maximum_filter
from haversine import haversine_vector, Unit

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
    angle_deg_madeira,
    radius_madeira,
    hm_m_model,
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


# =============================================================================
# region import data

################################ import tracked vortices
tracked_rvor = tb.open_file(
    'scratch/rvorticity/rvor_track/identified_transformed_rvor_20100803_09_track_4p_4n.h5',
    mode="r")
group_name = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']

################################ calculate distance to Madeira
distance2madeira = []
for j in range(len(group_name)):
    # j=0
    iexp = tracked_rvor.root[group_name[j]]
    center_lat = iexp.extracted_vortex_info.cols.center_lat[:]
    center_lon = iexp.extracted_vortex_info.cols.center_lon[:]
    jdistance2madeira = haversine_vector(
        [(center_madeira[1], center_madeira[0])],
        [tuple(i) for i in zip(center_lat, center_lon)],
        Unit.KILOMETERS, comb=True
    )
    distance2madeira.append(jdistance2madeira)


################################ plot decay of vortex size

################ extract size and distance info
pdistance = np.array([])
psize = np.array([])
ndistance = np.array([])
nsize = np.array([])
for j in range(len(group_name)):
    # j = 0
    jdistance = distance2madeira[j][:, 0]
    jsize = tracked_rvor.root[group_name[j]].extracted_vortex_info.cols.size[:]
    if j < 4:
        pdistance = np.concatenate((pdistance, jdistance))
        psize = np.concatenate((psize, jsize))
    else:
        ndistance = np.concatenate((ndistance, jdistance))
        nsize = np.concatenate((nsize, jsize))

# sort
psize = psize[np.argsort(pdistance)]
pdistance = pdistance[np.argsort(pdistance)]
nsize = nsize[np.argsort(ndistance)]
ndistance = ndistance[np.argsort(ndistance)]


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

################ plot size and distance
ax.plot(pdistance, psize, '.', color='red', markersize=2.5)
ax.plot(ndistance, nsize, '.', color='blue', markersize=2.5)

################ fit a exp line to them
from scipy.optimize import curve_fit
def func(x, a, b):
    return (a * x + b)
popt, pcov = curve_fit(func, pdistance, psize)
nopt, ncov = curve_fit(func, ndistance, nsize)

################ plot fitted line
plt_line1 = ax.plot(pdistance, func(pdistance, *popt), '.-',
                    color='r', linestyle='dashed', linewidth=0.5, markersize=0)
plt_line2 = ax.plot(ndistance, func(ndistance, *nopt), '.-',
                    color='b', linestyle='dashed', linewidth=0.5, markersize=0)

ax.set_ylim(0, 1650)
ax.set_yticks(np.arange(0, 1610, 200))
ax.set_yticklabels(np.arange(0, 1610, 200, dtype='int'))
ax.set_ylabel('Vortex size [$km^2$]')

ax.set_xlim(0, 640)
ax.set_xticks(np.arange(0, 640, 100))
ax.set_xticklabels(np.arange(0, 640, 100, dtype='int'))
ax.set_xlabel('Distance to Madeira center [$km$]')

ax_legend = ax.legend(
    [plt_line1[0], plt_line2[0]], ['Positive vortices', 'Negative vortices', ],
    loc='upper right', frameon=False, ncol=1, handlelength=1.5)
for i in range(len(ax_legend.get_lines())):
    # ax_legend.get_lines()[i].set_linewidth(1)
    ax_legend.legendHandles[i]._legmarker.set_markersize(2.5)

fig.subplots_adjust(left=0.18, right=0.99, bottom=0.14, top=0.99)
fig.savefig(
    'figures/09_decades_vortex/09_05_vortex_track/9_5_1.1 decay of vortex size.png', dpi=600)


tracked_rvor.close()

'''
# def func(x, a, b, c):
#     return (a * np.exp(-b * x) + c)

################ plot mean and peak magnitude
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
for j in range(len(group_name)):
    # j = 0
    ax.plot(
        distance2madeira[j][:, 0],
        tracked_rvor.root[
            group_name[j]].extracted_vortex_info.cols.peak_magnitude[:],
        '.-',
        color='black',
        linestyle='dashed', linewidth=0.5, markersize=2.5,
    )
    ax.plot(
        distance2madeira[j][:, 0],
        tracked_rvor.root[
            group_name[j]].extracted_vortex_info.cols.mean_magnitude[:],
        '.-',
        # color='black',
        linestyle='dashed', linewidth=0.5, markersize=2.5,
    )
    ################ plot vortex size - distance
    # ax.plot(
    #     jdistance, jsize, '.', color=col, markersize=2.5,
    #     # linestyle=ls, linewidth=0.5,
    # )
fig.savefig(
    'figures/09_decades_vortex/09_05_vortex_track/9_5_1.0 decay of vortex mean and peak rvor.png', dpi=600)
'''
# endregion
# =============================================================================

