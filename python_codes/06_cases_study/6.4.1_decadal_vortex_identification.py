

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


smaller_domain = True
iyear = 4
# =============================================================================
# region import data

# years[iyear]

################################ 100m vorticity
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values
lon = rvor_100m.lon[80:920, 80:920].values
lat = rvor_100m.lat[80:920, 80:920].values

################################ model topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values

################################ parameters setting
grid_size = 1.2  # in km^2
median_filter_size = 3
maximum_filter_size = 50

min_rvor = 3.
min_max_rvor = 4.
min_size = 100.
min_size_theta = 450.
min_size_dir = 450
min_size_dir1 = 900
max_dir = 30
max_dir1 = 40
max_dir2 = 50
max_distance2radius = 5


################################ original simulation to calculate surface theta
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')
))

################################ create a mask for Madeira
if smaller_domain:
    polygon = [
        (390, 0), (390, 320), (260, 580),
        (380, 839), (839, 839), (839, 0), ]
else:
    polygon = [
        (300, 0), (390, 320), (260, 580),
        (380, 839), (839, 839), (839, 0), ]

poly_path = Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)

################################ import wind
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

# endregion
# =============================================================================


# =============================================================================
# region decadal vortex identification

reject_info = False
istart = 0
ifinal = len(time)
if smaller_domain:
    outputfile = \
        "scratch/rvorticity/rvor_identify/re_identify/decades_past_sd/identified_transformed_rvor_20" + \
        years[iyear] + ".h5"
else:
    outputfile = \
        "scratch/rvorticity/rvor_identify/re_identify/decades_past/identified_transformed_rvor_20" + \
        years[iyear] + ".h5"

# time[istart: ifinal]
# orig_simulation_f[[istart,ifinal - 1]]

################################ create hdf5 file using PyTable

ilat = lon.shape[0]
ilon = lon.shape[1]

class VortexInfo(tb.IsDescription):
    vortex_count = tb.Int8Col()
    is_vortex = tb.Int8Col(shape=(ilat, ilon))
    vortex_indices = tb.Int8Col(shape=(ilat, ilon))

class VortexDeInfo(tb.IsDescription):
    time_index = tb.Int64Col()
    vortex_index = tb.Int8Col()
    center_lat = tb.Float32Col()
    center_lon = tb.Float32Col()
    size = tb.Float64Col()
    radius = tb.Float64Col()
    distance2radius = tb.Float64Col()
    peak_magnitude = tb.Float64Col()
    mean_magnitude = tb.Float64Col()
    mean_wind_u = tb.Float64Col()
    mean_wind_v = tb.Float64Col()
    angle = tb.Float32Col()

identified_rvor = tb.open_file(outputfile, mode="w")

################################ create group and table to store results

############ create group
experiment = identified_rvor.create_group('/', 'exp1', 'Experiment 1')

############ create arrays
identified_rvor.create_array(experiment, 'lon', lon, 'longitude')
identified_rvor.create_array(experiment, 'lat', lat, 'latitude')

############ assign attributes
experiment._v_attrs.min_rvor = min_rvor
experiment._v_attrs.min_max_rvor = min_max_rvor
experiment._v_attrs.min_size = min_size
experiment._v_attrs.min_size_theta = min_size_theta
experiment._v_attrs.min_size_dir = min_size_dir
experiment._v_attrs.min_size_dir1 = min_size_dir1
experiment._v_attrs.max_dir = max_dir
experiment._v_attrs.max_dir1 = max_dir1
experiment._v_attrs.max_dir2 = max_dir2
experiment._v_attrs.max_distance2radius = max_distance2radius
experiment._v_attrs.reject_info = reject_info
experiment._v_attrs.median_filter_size = median_filter_size
experiment._v_attrs.maximum_filter_size = maximum_filter_size

############ create tables
vortex_info = identified_rvor.create_table(
    experiment, 'vortex_info', VortexInfo, 'Vortex Information')
vortex_de_info = identified_rvor.create_table(
    experiment, 'vortex_de_info', VortexDeInfo, 'Vortex Detail Information')
############ extract rows
vortex_info_r = vortex_info.row
vortex_de_info_r = vortex_de_info.row
if reject_info:
    rejected_vortex_info = identified_rvor.create_table(
        experiment, 'rejected_vortex_info', VortexInfo,
        'Rejected vortex Information')
    rejected_vortex_de_info = identified_rvor.create_table(
        experiment, 'rejected_vortex_de_info', VortexDeInfo,
        'Rejected vortex Detail Information')
    ############ extract rows
    rejected_vortex_info_r = rejected_vortex_info.row
    rejected_vortex_de_info_r = rejected_vortex_de_info.row


for i in np.arange(istart, ifinal):  # np.arange(istart, istart + 4):  #
    begin_time = datetime.datetime.now()
    
    # i = istart
    ######## import relative vorticity and wind
    rvor = rvor_100m.relative_vorticity[i, 80:920, 80:920].values * 10**4
    wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
    wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
    
    ######## import theta
    orig_simulation = xr.open_dataset(orig_simulation_f[i])
    pres = orig_simulation.PS[0, 80:920, 80:920].values
    tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
    theta = tem2m * (p0sl/pres)**(r/cp)
    
    ######## wavelet transform
    coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
    n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
    rvorticity = rec_rvor
    original_rvorticity = rvor
    
    ######## vortex identification
    if reject_info:
        (vortices, is_vortex, vortices_count, vortex_indices, theta_anomalies,
         rejected_vortices, rejected_is_vortex, rejected_vortices_count,
         rejected_vortex_indices) = vortex_identification1(
             rvorticity, lat, lon, model_topo, theta, wind_u, wind_v,
             center_madeira, poly_path, madeira_mask,
             min_rvor, min_max_rvor, min_size, min_size_theta,
             min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
             max_distance2radius,
             original_rvorticity=original_rvorticity, reject_info=True,
             grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
         )
    else:
        (vortices, is_vortex, vortices_count, vortex_indices,
         theta_anomalies) = vortex_identification1(
             rvorticity, lat, lon, model_topo, theta, wind_u, wind_v,
             center_madeira, poly_path, madeira_mask,
             min_rvor, min_max_rvor, min_size, min_size_theta,
             min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
             max_distance2radius,
             original_rvorticity=original_rvorticity, reject_info=False,
             grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
        )
    
    ######## store vortex information
    vortex_info_r['is_vortex'] = is_vortex
    vortex_info_r['vortex_count'] = vortices_count
    vortex_info_r['vortex_indices'] = vortex_indices
    vortex_info_r.append()
    
    if reject_info:
        rejected_vortex_info_r['is_vortex'] = rejected_is_vortex
        rejected_vortex_info_r['vortex_count'] = rejected_vortices_count
        rejected_vortex_info_r['vortex_indices'] = rejected_vortex_indices
        rejected_vortex_info_r.append()
    
    ######## store vortex detail information
    for j in range(len(vortices)):
        vortex_de_info_r['time_index'] = i
        vortex_de_info_r['vortex_index'] = j
        vortex_de_info_r['center_lat'] = vortices[j]['center_lat']
        vortex_de_info_r['center_lon'] = vortices[j]['center_lon']
        vortex_de_info_r['size'] = vortices[j]['size']
        vortex_de_info_r['radius'] = vortices[j]['radius']
        vortex_de_info_r['distance2radius'] = vortices[j]['distance2radius']
        vortex_de_info_r['peak_magnitude'] = vortices[j]['peak_magnitude']
        vortex_de_info_r['mean_magnitude'] = vortices[j]['mean_magnitude']
        vortex_de_info_r['mean_wind_u'] = vortices[j]['mean_wind_u']
        vortex_de_info_r['mean_wind_v'] = vortices[j]['mean_wind_v']
        vortex_de_info_r['angle'] = vortices[j]['angle']
        vortex_de_info_r.append()
    
    if reject_info:
        for j in range(len(rejected_vortices)):
            rejected_vortex_de_info_r['time_index'] = i
            rejected_vortex_de_info_r['vortex_index'] = j
            rejected_vortex_de_info_r['center_lat'] = \
                rejected_vortices[j]['center_lat']
            rejected_vortex_de_info_r['center_lon'] = \
                rejected_vortices[j]['center_lon']
            rejected_vortex_de_info_r['size'] = rejected_vortices[j]['size']
            rejected_vortex_de_info_r['radius'] = rejected_vortices[j]['radius']
            rejected_vortex_de_info_r['distance2radius'] = \
                rejected_vortices[j]['distance2radius']
            rejected_vortex_de_info_r['peak_magnitude'] = \
                rejected_vortices[j]['peak_magnitude']
            rejected_vortex_de_info_r['mean_magnitude'] = \
                rejected_vortices[j]['mean_magnitude']
            rejected_vortex_de_info_r['mean_wind_u'] = \
                rejected_vortices[j]['mean_wind_u']
            rejected_vortex_de_info_r['mean_wind_v'] = \
                rejected_vortices[j]['mean_wind_v']
            rejected_vortex_de_info_r['angle'] = \
                rejected_vortices[j]['angle']
            rejected_vortex_de_info_r.append()
    
    print(str(i) + ' ' + str(datetime.datetime.now() - begin_time))

vortex_info.flush()
vortex_de_info.flush()
if reject_info:
    rejected_vortex_info.flush()
    rejected_vortex_de_info.flush()

identified_rvor.close()


'''
# check
identified_rvor = tb.open_file(
    "scratch/rvorticity/rvor_identify/re_identify/decades_past/identified_transformed_rvor_20" + years[iyear] + ".h5",
    mode="r")
experiment = identified_rvor.root.exp1

# istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
istart = 0
i = 3

######## relative vorticity and wind
rvor = rvor_100m.relative_vorticity[istart + i, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[istart + i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[istart + i, 80:920, 80:920].values

######## theta
orig_simulation = xr.open_dataset(orig_simulation_f[istart + i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ wavelet transform
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
rvorticity = rec_rvor
original_rvorticity = rvor

if experiment._v_attrs.reject_info:
    (vortices1, is_vortex1, vortices_count1, vortex_indices1, theta_anomalies1,
     rejected_vortices1, rejected_is_vortex1, rejected_vortices_count1,
     rejected_vortex_indices1) = vortex_identification1(
         rvorticity, experiment.lat.read(), experiment.lon.read(), model_topo,
         theta, wind_u, wind_v,
         center_madeira, poly_path, madeira_mask,
         experiment._v_attrs.min_rvor, experiment._v_attrs.min_max_rvor,
         experiment._v_attrs.min_size, experiment._v_attrs.min_size_theta,
         experiment._v_attrs.min_size_dir, experiment._v_attrs.min_size_dir1,
         experiment._v_attrs.max_dir, experiment._v_attrs.max_dir1,
         experiment._v_attrs.max_dir2, experiment._v_attrs.max_distance2radius,
         original_rvorticity = original_rvorticity,
         reject_info = experiment._v_attrs.reject_info,
         grid_size=1.2,
         median_filter_size = experiment._v_attrs.median_filter_size,
         maximum_filter_size = experiment._v_attrs.maximum_filter_size,
         )
else:
    (vortices1, is_vortex1, vortices_count1, vortex_indices1,
     theta_anomalies1) = vortex_identification1(
         rvorticity, experiment.lat.read(), experiment.lon.read(), model_topo,
         theta, wind_u, wind_v,
         center_madeira, poly_path, madeira_mask,
         experiment._v_attrs.min_rvor, experiment._v_attrs.min_max_rvor,
         experiment._v_attrs.min_size, experiment._v_attrs.min_size_theta,
         experiment._v_attrs.min_size_dir, experiment._v_attrs.min_size_dir1,
         experiment._v_attrs.max_dir, experiment._v_attrs.max_dir1,
         experiment._v_attrs.max_dir2, experiment._v_attrs.max_distance2radius,
         original_rvorticity = original_rvorticity,
         reject_info = experiment._v_attrs.reject_info,
         grid_size=1.2,
         median_filter_size = experiment._v_attrs.median_filter_size,
         maximum_filter_size = experiment._v_attrs.maximum_filter_size,
         )

# compare
(is_vortex1 == experiment.vortex_info.cols.is_vortex[i]).all()
vortices_count1 == experiment.vortex_info.cols.vortex_count[i]
(vortex_indices1 == experiment.vortex_info.cols.vortex_indices[i]).all()


vortices1[-1]['center_lat'] == experiment.vortex_de_info.cols.center_lat[-1]
vortices1[-2]['center_lon'] == experiment.vortex_de_info.cols.center_lon[-2]
vortices1[-2]['peak_magnitude'] == experiment.vortex_de_info.cols.peak_magnitude[-2]
vortices1[-2]['mean_magnitude'] == experiment.vortex_de_info.cols.mean_magnitude[-2]
vortices1[-2]['size'] == experiment.vortex_de_info.cols.size[-2]
vortices1[-2]['radius'] == experiment.vortex_de_info.cols.radius[-2]
vortices1[-2]['distance2radius'] == experiment.vortex_de_info.cols.distance2radius[-2]

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': '2010-08-00 0000', 'time_location': [-23, 34],},
    )
for j in range(experiment.vortex_info.cols.vortex_count[i]):
    ax.text(
        [row['center_lon'] for row in experiment.vortex_de_info.where(
            '(time_index == ' + str(istart + i) +
            ') & (vortex_index == ' + str(j) + ')')][0],
        [row['center_lat']
         for row in experiment.vortex_de_info.where(
            '(time_index == ' + str(istart + i) +
            ') & (vortex_index == ' + str(j) + ')')][0],
        str(j), color = 'm', size = 8, fontweight='bold')
ax.contour(lon, lat, experiment.vortex_info.cols.is_vortex[i],
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.2, linestyles='solid'
           )
ax.contour(lon, lat, experiment.rejected_vortex_info.cols.is_vortex[i],
           colors='red', levels=np.array([-0.5, 0.5]),
           linewidths=0.1, linestyles='solid'
           )
fig.savefig('figures/00_test/trial.png', dpi=600)
plt.close('all')
'''
# endregion
# =============================================================================



