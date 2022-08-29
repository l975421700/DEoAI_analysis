

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
import seaborn as sns

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
from scipy import ndimage
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py

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


smaller_domain = False
# =============================================================================
# region import data

if smaller_domain:
    inputfile = 'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015_sd.pkl'
    outputfile1 = 'figures/09_decades_vortex/09_07_smaller_domain/9_7.1.3_monthly_madeira_vortex_count_boxplot.png'
    outputfile2 = 'figures/09_decades_vortex/09_07_smaller_domain/9_7.1.4_annual_madeira_vortex_count_boxplot.png'
else:
    inputfile = 'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015.pkl'
    outputfile1 = 'figures/09_decades_vortex/9.0.5 monthly madeira vortex count_boxplot.png'
    outputfile2 = 'figures/09_decades_vortex/9.0.6 annual madeira vortex count_boxplot.png'

decades_vortex_count = pd.read_pickle(
    inputfile).astype('int64').rename('count')
# np.log2(len(decades_vortex_count))

date = pd.date_range("2010-01-01", '2010-12-31', freq="M")
time = pd.date_range("2010-01-01-00", '2010-12-31-23', freq="D")
decades_vortex_count_daily = decades_vortex_count.resample('1D').sum()
decades_vortex_count_monthly = decades_vortex_count.resample('1M').sum()
decades_vortex_count_annual = decades_vortex_count.resample('1Y').sum()

# endregion
# =============================================================================


# =============================================================================
# region wavelet transform hourly vortex count

################ wavelet denoise
coeffs = pywt.wavedec(decades_vortex_count.values, 'haar', mode='periodic')

n_0ratio = 0.5
# flatten the coeffients
coeffs_nd, coeff_slices = pywt.coeffs_to_array(coeffs, padding=0, axes=None)

# normalized square modulus of the ith element of the signal
nsm_signal = decades_vortex_count.values ** 2 / \
    np.sum(decades_vortex_count.values**2)
# The entropy
entropy = -np.sum(nsm_signal[nsm_signal != 0] *
                  np.log(nsm_signal[nsm_signal != 0]))
# the number of significant coefficient
n_0 = np.int(np.e ** entropy * n_0ratio)
n_0 = np.int(len(decades_vortex_count) * 0.05)

coeffs_nd[abs(coeffs_nd) <= np.sort(abs(coeffs_nd))[-n_0-1]] = 0

rec_coeffs_nd = pywt.array_to_coeffs(
    coeffs_nd, coeff_slices, output_format='wavedec')

rec_signal = pywt.waverec(rec_coeffs_nd, 'haar', mode='periodic')
pd_rec_signal = pd.Series(
    data=rec_signal, index=decades_vortex_count.index)

################ plot original and reconstructed hourly count in 2010

pd_rec_signal_daily = pd_rec_signal.resample('1D').sum()

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_hcount1, = ax.plot(
    time,
    decades_vortex_count_daily.loc[time], linewidth=0.5, color='black'
)
plt_hcount2, = ax.plot(
    time,
    pd_rec_signal_daily.loc[time], linewidth=0.5, color='red'
)

ax.set_xlim(time[0] - np.timedelta64(1, 'D'),
            time[-1] + np.timedelta64(1, 'D'))
ax.set_xticks(date)
ax.set_xticklabels(month, size=8)
ax.set_xlabel('Year 2010', size=10)

ax.set_yticks(np.arange(0, 110, 20))
ax.set_yticklabels(np.arange(0, 110, 20), size=8)
ax.set_ylabel("Daily number of vortices [#]", size=10)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.97)
fig.savefig(
    'figures/09_decades_vortex/9.0.4 hourly madeira vortex count and wavelet transformed.png',
    dpi=600)

# endregion
# =============================================================================


# =============================================================================
# region boxplot for each month and each year
decades_vortex_count_monthly = decades_vortex_count_monthly.to_frame()
decades_vortex_count_monthly['month'] = decades_vortex_count_monthly.index.month
decades_vortex_count_monthly['year'] = decades_vortex_count_monthly.index.year

################################ monthly boxplot

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_box = sns.boxplot(
    data=decades_vortex_count_monthly, x='month', y='count', ax=ax,
    order=np.arange(1, 13, 1), fliersize=0, linewidth=0.8, width = 0.8,)
for i in range(len(plt_box.artists)):
    plt_box.artists[i].set_facecolor('white')

plt_swarm = sns.swarmplot(
    data=decades_vortex_count_monthly, x='month', y='count', ax=ax, hue='year',
    order=np.arange(1, 13, 1), hue_order=np.arange(2006, 2016, 1), size=3,
    palette=sns.color_palette(
        'viridis_r', len(np.unique(decades_vortex_count_monthly['year']))),)
plt_legend = ax.legend(
    title=None, labelspacing=0.2, fontsize=8, handletextpad=0.2,
    markerscale=0.3,)

ax.set_xticks(np.arange(0, 12, 1))
ax.set_xticklabels(month, size=8)
ax.set_xlabel("Months from 2006 to 2015", size=10)

ax.set_ylim(0, 1500)
ax.set_yticks(np.arange(0, 1501, 300))
ax.set_yticklabels(np.arange(0, 1501, 300), size=8)
ax.set_ylabel("Monthly count of vortices [#]", size=10)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.15, right=0.99, bottom=0.14, top=0.97)
fig.savefig(outputfile1, dpi=600)


################################ annual boxplot
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_box = sns.boxplot(
    data=decades_vortex_count_monthly, x='year', y='count', ax=ax,
    order=np.arange(2006, 2016, 1), fliersize=0, linewidth=0.8, width=0.8,)
for i in range(len(plt_box.artists)):
    plt_box.artists[i].set_facecolor('white')

plt_swarm = sns.swarmplot(
    data=decades_vortex_count_monthly, x='year', y='count', ax=ax, hue='month',
    order=np.arange(2006, 2016, 1), hue_order=np.arange(1, 13, 1), size=3,
    palette=sns.color_palette(
        "viridis_r", len(np.unique(decades_vortex_count_monthly['month']))),)
plt_legend = ax.legend(
    title=None, labelspacing=0.2, fontsize=8, handletextpad=0.2,
    markerscale=0.3, loc='upper right', ncol=2, columnspacing=1,)

# ax.set_xlim(2006, 2015)
# ax.set_xticks(np.arange(2006, 2016, 1))
ax.set_xticklabels(years, size=8)
ax.set_xlabel("Years", size=10)

ax.set_ylim(0, 1500)
ax.set_yticks(np.arange(0, 1501, 300))
ax.set_yticklabels(np.arange(0, 1501, 300), size=8)
ax.set_ylabel("Monthly count of vortices [#]", size=10)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.15, right=0.99, bottom=0.14, top=0.97)
fig.savefig(outputfile2, dpi=600)


# endregion
# =============================================================================




