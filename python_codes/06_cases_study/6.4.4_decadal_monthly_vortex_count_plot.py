

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
# region import identified vortices

if smaller_domain:
    identified_rvor_f = sorted(glob.glob(
        'scratch/rvorticity/rvor_identify/re_identify/decades_past_sd/identified_transformed_rvor_20*.h5'))
    outputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015_sd.pkl"
else:
    identified_rvor_f = sorted(glob.glob(
        'scratch/rvorticity/rvor_identify/re_identify/decades_past/identified_transformed_rvor_20*.h5'))
    outputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015.pkl"

decades_vortex_count = pd.Series(dtype='int8')

for i in range(len(identified_rvor_f)):
    begin_time = datetime.datetime.now()
    print(begin_time)
    # i = 0
    
    time = pd.date_range(
        "20" + years[i] + "-01-01-00",
        '20' + years[i] + '-12-31-23',
        freq="60min")
    
    # read in data
    identified_rvor = tb.open_file(identified_rvor_f[i], mode="r")
    experiment = identified_rvor.root.exp1
    hourly_vortex_count = experiment.vortex_info.cols.vortex_count[:]
    
    # filer isolated vortices
    previous_count = np.concatenate((np.array([0]), hourly_vortex_count[:-1]))
    next_count = np.concatenate((hourly_vortex_count[1:], np.array([0]), ))
    isolated_hour = np.vstack(((hourly_vortex_count > 0),
                               (previous_count == 0),
                               (next_count == 0))).all(axis=0)
    filtered_hourly_vortex_count = hourly_vortex_count.copy()
    filtered_hourly_vortex_count[isolated_hour] = 0
    
    # append to time series
    pd_hourly_vortex_count = pd.Series(
        data=filtered_hourly_vortex_count, index=time)
    decades_vortex_count = decades_vortex_count.append(pd_hourly_vortex_count)
    
    identified_rvor.close()
    print(str(i) + "/" + str(len(identified_rvor_f) - 1) + "   " +
          str(datetime.datetime.now() - begin_time))

decades_vortex_count.to_pickle(outputfile)


'''
# check

check_decades_vortex_count = pd.read_pickle(
    'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015.pkl')
i=4

(check_decades_vortex_count.loc[time].values == filtered_hourly_vortex_count).all()


decades_vortex_count.to_csv(
    'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015.csv', header = False
)
check_decades_vortex_count = pd.read_csv(
    'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015.csv',
    index_col = 0, header=None, squeeze=True
)
'''

# endregion
# =============================================================================


# =============================================================================
# region decadal monthly vortex count plot
if smaller_domain:
    decades_vortex_count = pd.read_pickle(
        'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015_sd.pkl')
    outputfile = 'figures/09_decades_vortex/09_07_smaller_domain/9_7.1.2_decadal_monthly_vortex_count_sd.png'
else:
    decades_vortex_count = pd.read_pickle(
        'scratch/rvorticity/rvor_identify/re_identify/decades_vortex_count2006_2015.pkl')
    outputfile = 'figures/09_decades_vortex/9.0.3 decadal monthly vortex count.png'

decades_vortex_count_daily = decades_vortex_count.resample('1D').sum()
decades_vortex_count_monthly = decades_vortex_count.resample('1M').sum()

################################ plot decadal daily vortex count

date = pd.date_range("2006-01-01", '2015-12-31', freq="Y")

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

# plt_dcount1, = ax.plot(
#     decades_vortex_count_daily.index,
#     decades_vortex_count_daily.values, linewidth=0.5, color='black'
# )

plt_mcound2, = ax.plot(
    decades_vortex_count_monthly.index,
    decades_vortex_count_monthly.values, linewidth=0.5, color='black'
)

ax.set_xlim(decades_vortex_count_daily.index[0] - np.timedelta64(30, 'D'),
            decades_vortex_count_daily.index[-1] + np.timedelta64(30, 'D'))
ax.set_xticks(date)
ax.set_xticklabels(years, size=8)
ax.set_xlabel('Years', size=10)

ax.set_ylim(0, 1500)
ax.set_yticks(np.arange(0, 1501, 300))
ax.set_yticklabels(np.arange(0, 1501, 300), size=8)
ax.set_ylabel("Monthly count of vortices [#]", size=10)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.97)
fig.savefig(outputfile, dpi=600)

'''
time = pd.date_range(
    "2010-01-01",
    '2010-12-31',
    freq="D")
'''
# endregion
# =============================================================================


