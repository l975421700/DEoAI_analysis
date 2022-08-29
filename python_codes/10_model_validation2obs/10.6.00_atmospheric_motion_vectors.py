

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
from scipy.ndimage import median_filter

from DEoAI_analysis.module.vortex_namelist import (
    correctly_identified
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
    center_madeira,
    angle_deg_madeira,
    radius_madeira,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    find_nearest_grid,
    rotate_wind,
    sig_coeffs,
    vortex_identification,
    vortex_identification1,
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    inversion_layer,
)


# endregion
# =============================================================================


# =============================================================================
# region plot amv

from pybufr_ecmwf.bufr import BUFRReader

file = 'data_source/eumetsat/amv/1509238-1of1/MSG2-SEVI-MSGAMVE-0100-0100-20100805104500.000000000Z-20100805110546-1509238.bfr'
# file = 'ascat_20100805_103000_metopa_19687_srv_o_063_ovw.l2_bufr'

bufr = BUFRReader(file)

# display number of messages
print("Number of messages: " + str(bufr.num_msgs))

names_units = []
for m, msg in enumerate(bufr):
    names_units.append(msg.get_names_and_units())

print('\n'.join(map(str, names_units[0][0])))


'''
for i, msg in enumerate(bufr):
    for j, msg_or_subset_data in enumerate(msg):
        if ((i==0) & (j==0)):
            data = msg_or_subset_data.data
            names = msg_or_subset_data.names
            units = msg_or_subset_data.units

'''
# endregion
# =============================================================================


