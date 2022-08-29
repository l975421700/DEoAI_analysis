

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
    correctly_reidentified
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
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    sig_coeffs,
    vortex_identification,
)


# endregion


# region data import and preprocessing
################################ 100m vorticity
iyear = 4
# years[iyear]
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values

inputfile = 'scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5'
identified_rvor = tb.open_file(inputfile, mode="r")
experiment = identified_rvor.root.exp1

# correctly_reidentified
# experiment._v_attrs
# experiment.lat.read()
# experiment.vortex_info.cols.vortex_count[:]


bool_identified = np.array([], dtype='int64')
bool_not_identified = np.array([], dtype='int64')
for k, (i, j) in enumerate(zip(
        experiment.vortex_de_info.cols.time_index[:],
        experiment.vortex_de_info.cols.vortex_index[:])):
    # print(str(k) + '   ' + str(i) + '   ' + str(j))
    if (j in (correctly_reidentified[str(time[i])[0:13]])):
        bool_identified = np.append(bool_identified, k)
    else:
        bool_not_identified = np.append(bool_not_identified, k)


'''
for indices in range(len(experiment.vortex_de_info.cols.time_index)):
    # indices = 42
    if (not indices in bool_identified):
        print(
            # 'time:' + str(experiment.vortex_de_info.cols.time_index[indices]) +
            # ' i:' +
            str(experiment.vortex_de_info.cols.vortex_index[indices]) +
            ' not in:' + str(correctly_reidentified[str(time[
                experiment.vortex_de_info.cols.time_index[indices]])[0:13]])
            )

for indices in range(len(experiment.vortex_de_info.cols.time_index)):
    # indices = 42
    if (indices in bool_not_identified):
        print(
            # 'time:' + str(experiment.vortex_de_info.cols.time_index[indices]) +
            # ' i:' +
            str(experiment.vortex_de_info.cols.vortex_index[indices]) +
            ' not in:' + str(correctly_reidentified[str(time[
                experiment.vortex_de_info.cols.time_index[indices]])[0:13]])
            )

# some small cells can result in very small distance2radius (< 28 grids)
# and very large ellipse_eccentricity ( < 12 grids)
np.max(experiment.rejected_vortex_de_info.cols.size[:][np.where(experiment.rejected_vortex_de_info.cols.distance2radius[:] < 2)[0]])

'''

'''
plt.hist(
    # experiment.vortex_de_info.cols.ellipse_eccentricity[:],
    # experiment.vortex_de_info.cols.distance2radius[:],
    # experiment.vortex_de_info.cols.mean_magnitude[:],
    # experiment.vortex_de_info.cols.peak_magnitude[:],
    # experiment.vortex_de_info.cols.size[:],
    # experiment.rejected_vortex_de_info.cols.ellipse_eccentricity[:][
    #     np.isfinite(
    #         experiment.rejected_vortex_de_info.cols.ellipse_eccentricity[:])],
    # experiment.rejected_vortex_de_info.cols.distance2radius[:],
    # experiment.rejected_vortex_de_info.cols.mean_magnitude[:],
    # experiment.rejected_vortex_de_info.cols.peak_magnitude[:],
    experiment.rejected_vortex_de_info.cols.size[:],
    bins=20)
plt.savefig('figures/00_test/trial.png', dpi=600)
plt.close('all')
'''

# endregion
# =============================================================================


# =============================================================================
# region plot vortices size

rejected_vortices_size = experiment.rejected_vortex_de_info.cols.size[:]
vortices_size = experiment.vortex_de_info.cols.size[:]
# np.max(np.concatenate((vortices_size, rejected_vortices_size)))

fig, ax = plt.subplots(1, 1, figsize=np.array([8, 8]) / 2.54)
n, bins, patches1 = ax.hist(
    rejected_vortices_size, color='dodgerblue', bins=np.arange(0, 4000, 10),
    alpha = 0.8,
    # density=True,
    )
n, bins, patches2 = ax.hist(
    vortices_size, color='red', bins=np.arange(0, 4000, 10),
    )
n, bins, patches3 = ax.hist(
    vortices_size[bool_not_identified], color='black', bins=np.arange(0, 4000, 10),
    )

plt_threshold = plt.axvline(x=100, color='gray', linestyle='--', lw=1)
plt.xscale('log')
plt.yscale('log')
ax.set_xlim(7, 5000)
ax.set_xticks([10, 100, 1000])
ax.set_xlabel('Vortex size [$km^2$]', labelpad=1)
ax.set_ylabel('Count of vortices [#]', labelpad=1)
ax.grid(True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
ax.legend([patches1[0], patches2[0], patches3[0], plt_threshold],
          ['Extracted vortices', 'Identified vortices',
           'Falsely identified vortices', r'$100 \; km^2$'],
          loc='lower center', frameon=False, ncol=2,
          bbox_to_anchor=(0.41, -0.4), handlelength=1,
          columnspacing=0.5,
          )
fig.subplots_adjust(left=0.15, right=0.99, bottom=0.26, top=0.98)
fig.savefig(
    'figures/09_decades_vortex/09_04_vortex_statistics/9_4_0.0 histogram of vortex size.png',
    dpi=600)
plt.close('all')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices mean magnitude

rejected_vortices_mean = \
    experiment.rejected_vortex_de_info.cols.mean_magnitude[:]
'''
np.sum(abs(experiment.rejected_vortex_de_info.cols.mean_magnitude[:]) < 3) / len(rejected_vortices_mean)

# avverage size 4 km^2
np.max(
    experiment.rejected_vortex_de_info.cols.size[:][
        np.where(
            abs(experiment.rejected_vortex_de_info.cols.mean_magnitude[:]) < 3
            )[0]])

# np.where((experiment.rejected_vortex_de_info.cols.size[:] == 110.39999999999999) & (abs(experiment.rejected_vortex_de_info.cols.mean_magnitude[:]) < 3))

# time_index: 5221, size: 110.39999999999999, mean_mag: 2.9653055391263625
'''
vortices_mean = experiment.vortex_de_info.cols.mean_magnitude[:]
# np.max(np.concatenate((vortices_mean, rejected_vortices_mean)))
hist_bins = np.arange(-54, 54, 0.25)


fig, ax = plt.subplots(1, 1, figsize=np.array([8, 8]) / 2.54)
n, bins, patches1 = ax.hist(
    rejected_vortices_mean, color='dodgerblue', bins=hist_bins,
    alpha = 0.8,
    )
n, bins, patches2 = ax.hist(
    vortices_mean, color='red', bins=hist_bins,
    )
n, bins, patches3 = ax.hist(
    vortices_mean[bool_not_identified], color='black', bins=hist_bins,
    )

plt.axvline(x=3, color='gray', linestyle='--', lw=1)
plt_threshold = plt.axvline(x=-3, color='gray', linestyle='--', lw=1)
plt.yscale('log')
ax.set_xlim(-40, 40)
ax.set_xticks(np.arange(-40, 40 + 1, 10))
ax.set_xlabel('Mean relative vorticity [$10^{-4}\;s^{-1}$]', labelpad=1)
ax.set_ylabel('Count of vortices [#]', labelpad=1)
ax.grid(True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
ax.legend([patches1[0], patches2[0], patches3[0], plt_threshold],
          ['Extracted vortices', 'Identified vortices',
           'Falsely identified vortices', r'$\pm \; 3 \times 10^{-4}\;s^{-1}$'],
          loc='lower center', frameon=False, ncol=2,
          bbox_to_anchor=(0.41, -0.4), handlelength=1,
          columnspacing=0.5,
          )
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.26, top=0.98)
fig.savefig(
    'figures/09_decades_vortex/09_04_vortex_statistics/9_4_0.1 histogram of vortex mean relative vorticity.png',
    dpi=600)
plt.close('all')


# endregion
# =============================================================================


# =============================================================================
# region plot vortices peak magnitude

rejected_vortices_peak = \
    experiment.rejected_vortex_de_info.cols.peak_magnitude[:]
vortices_peak = experiment.vortex_de_info.cols.peak_magnitude[:]
# stats.describe(np.concatenate((vortices_peak, rejected_vortices_peak)))
hist_bins = np.arange(-150, 150, 1)


fig, ax = plt.subplots(1, 1, figsize=np.array([8, 8]) / 2.54)
n, bins, patches1 = ax.hist(
    rejected_vortices_peak, color='dodgerblue', bins=hist_bins,
    alpha=0.8,
)
n, bins, patches2 = ax.hist(
    vortices_peak, color='red', bins=hist_bins,
)
n, bins, patches3 = ax.hist(
    vortices_peak[bool_not_identified], color='black', bins=hist_bins,
)

plt.axvline(x=4, color='gray', linestyle='--', lw=1)
plt_threshold = plt.axvline(x=-4, color='gray', linestyle='--', lw=1)
plt.yscale('log')
ax.set_xlim(-150, 150)
ax.set_xticks(np.arange(-150, 150 + 1, 50))
# plt.xscale('symlog')
# ax.set_xticks([-100, -10, 0, 10, 100])
ax.set_xlabel('Peak relative vorticity [$10^{-4}\;s^{-1}$]', labelpad=1)
ax.set_ylabel('Count of vortices [#]', labelpad=1)
ax.grid(True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
ax.legend([patches1[0], patches2[0], patches3[0], plt_threshold],
          ['Extracted vortices', 'Identified vortices',
           'Falsely identified vortices',
           r'$\pm \; 4 \times 10^{-4}\;s^{-1}$'],
          loc='lower center', frameon=False, ncol=2,
          bbox_to_anchor=(0.41, -0.4), handlelength=1,
          columnspacing=0.5,
          )
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.26, top=0.98)
fig.savefig(
    'figures/09_decades_vortex/09_04_vortex_statistics/9_4_0.2 histogram of vortex peak relative vorticity.png',
    dpi=600)
plt.close('all')

# endregion
# =============================================================================


# =============================================================================
# region plot largest distance to nominal radius


rejected_distance2radius = \
    experiment.rejected_vortex_de_info.cols.distance2radius[:]
distance2radius = experiment.vortex_de_info.cols.distance2radius[:]
# stats.describe(np.concatenate((distance2radius, rejected_distance2radius)))
hist_bins = np.arange(0, 15, 0.1)


fig, ax = plt.subplots(1, 1, figsize=np.array([8, 8]) / 2.54)
n, bins, patches1 = ax.hist(
    rejected_distance2radius, color='dodgerblue', bins=hist_bins,
    alpha=0.8,
)
n, bins, patches2 = ax.hist(
    distance2radius, color='red', bins=hist_bins,
)
n, bins, patches3 = ax.hist(
    distance2radius[bool_not_identified], color='black', bins=hist_bins,
)

plt_threshold = plt.axvline(x=5, color='gray', linestyle='--', lw=1)
plt.yscale('log')
ax.set_xlim(1, 15)
ax.set_xticks(np.arange(0, 15 + 1, 3))
ax.set_xlabel('Ratio of largest distance to nominal radius', labelpad=1)
ax.set_ylabel('Count of vortices [#]', labelpad=1)
ax.grid(True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
ax.legend([patches1[0], patches2[0], patches3[0], plt_threshold],
          ['Extracted vortices', 'Identified vortices',
           'Falsely identified vortices',
           r'$5$'],
          loc='lower center', frameon=False, ncol=2,
          bbox_to_anchor=(0.41, -0.4), handlelength=1,
          columnspacing=0.5,
          )
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.26, top=0.98)
fig.savefig(
    'figures/09_decades_vortex/09_04_vortex_statistics/9_4_0.3 histogram of largest distance to nominal radius.png',
    dpi=600)
plt.close('all')




# endregion
# =============================================================================


# =============================================================================
# region plot angle between mean winds within vortex and Madeira centre


rejected_angle = experiment.rejected_vortex_de_info.cols.angle[:]
angle = experiment.vortex_de_info.cols.angle[:]
# stats.describe(np.concatenate((angle, rejected_angle)))
hist_bins = np.arange(0, 180 + 0.1, 0.5)


fig, ax = plt.subplots(1, 1, figsize=np.array([8, 8]) / 2.54)
n, bins, patches1 = ax.hist(
    rejected_angle, color='dodgerblue', bins=hist_bins,
    alpha=0.8,
)
n, bins, patches2 = ax.hist(
    angle, color='red', bins=hist_bins,
)
n, bins, patches3 = ax.hist(
    angle[bool_not_identified], color='black', bins=hist_bins,
)

plt_threshold = plt.axvline(x=30, color='gray', linestyle='--', lw=1)
plt_threshold = plt.axvline(x=40, color='gray', linestyle='--', lw=1)
plt_threshold = plt.axvline(x=50, color='gray', linestyle='--', lw=1)


plt.yscale('log')
ax.set_xlim(0, 180)
ax.set_xticks(np.arange(0, 180, 30))
ax.set_xlabel('Angle [°]', labelpad=1)
ax.set_ylabel('Count of vortices [#]', labelpad=1)
ax.grid(True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
ax.legend([patches1[0], patches2[0], patches3[0], plt_threshold],
          ['Extracted vortices', 'Identified vortices',
           'Falsely identified vortices',
           r'$30°, 40°, 50°$'],
          loc='lower center', frameon=False, ncol=2,
          bbox_to_anchor=(0.41, -0.4), handlelength=1,
          columnspacing=0.5,
          )
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.26, top=0.98)
fig.savefig(
    'figures/09_decades_vortex/09_04_vortex_statistics/9_4_0.4 histogram of angle.png',
    dpi=600)
plt.close('all')




# endregion
# =============================================================================


