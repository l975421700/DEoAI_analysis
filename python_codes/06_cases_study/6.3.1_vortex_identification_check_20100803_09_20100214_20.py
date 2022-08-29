

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


# iyear = 4
iyear = 9
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
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

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
# max_dir = 30
# max_dir1 = 60
# max_dir2 = 90
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
polygon = [
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
]
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
# region re identify vortex in 20100803_09 and 201002_14_20

reject_info = False
# istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-08-09T08:00:00.000000000'))[0][0]
# outputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09.h5"
# outputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5"
istart = np.where(time == np.datetime64('2010-02-14T00:00:00.000000000'))[0][0]
ifinal = np.where(time == np.datetime64('2010-02-21T00:00:00.000000000'))[0][0]
# outputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20.h5"
outputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20_stricter_dir.h5"
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
experiment._v_attrs.time = time[istart:ifinal]

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


for i in np.arange(istart, ifinal):  # np.arange(istart+73, istart + 75):  #
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
    "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09.h5",
    mode="r")
experiment = identified_rvor.root.exp1

istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
i = 131

identified_rvor = tb.open_file(
    "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20.h5",
    mode="r")
experiment = identified_rvor.root.exp1

istart = np.where(time == np.datetime64('2010-02-14T00:00:00.000000000'))[0][0]
i = 167

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


# =============================================================================
# region define functions to find and plot local maximum

def find_plot_maxmin_points(
        lon, lat, data, ax, extrema, nsize, pointsize, pointcolor):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    ax: axis to plot
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    
    color: colors points
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, \
        minimum_filter, median_filter
    import numpy as np
    
    # data = gaussian_filter(data, sigma=3.0)
    data = median_filter(data, size=3)
    # add dummy variables to avoid identical neighbor values
    dummy = np.random.normal(0, 0.000001, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    
    for i in range(len(mxy)):
        # 1st criterion
        # criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
        #              (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        criteria1 = True
        if criteria1:
            ax.scatter(
                lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                s=pointsize, c=pointcolor,
            )
            # ax.text(
            #     lon[mxx[i]], lat[mxy[i]], symbol,
            #     color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
            #     horizontalalignment='center', verticalalignment='center')
        # if (criteria1 & plotValue):
        #     ax.text(
        #         lon[mxx[i]], lat[mxy[i]],
        #         '\n' + str(np.int(data[mxy[i], mxx[i]])),
        #         color=color, clip_on=True, clip_box=ax.bbox,
        #         fontweight='bold', horizontalalignment='center',
        #         verticalalignment='top')

'''
lon
lat
data = theta
extrema = 'max'
nsize = 50
'''
# endregion
# =============================================================================


# =============================================================================
# region plot reidentified rvor

# decade_plot = False
# istart = np.where(time == np.datetime64('2010-08-03T20:00'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-08-09T08:00'))[0][0]
# inputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09.h5"
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_1.0 Re_Identified transformed rvor and local theta anomaly_20100803_09.pdf'
# inputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5"
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_1.2 Re_Identified transformed rvor and local theta anomaly_20100803_09_stricter_dir.pdf'

# istart = np.where(time == np.datetime64('2010-02-14T00:00'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-02-21T00:00'))[0][0]
# inputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20.h5"
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_1.1 Re_Identified transformed rvor and local theta anomaly_20100214_20.pdf'
# inputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20_stricter_dir.h5"
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_1.3 Re_Identified transformed rvor and local theta anomaly_20100214_20_stricter_dir.pdf'

decade_plot = True
istart = np.where(time == np.datetime64('2015-03-01T00:00'))[0][0]
ifinal = np.where(time == np.datetime64('2015-03-31T23:00'))[0][0]
inputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_past/identified_transformed_rvor_2015.h5"
outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_1.4 Re_Identified transformed rvor and local theta anomaly_201503.pdf'

identified_rvor = tb.open_file(inputfile, mode="r")
experiment = identified_rvor.root.exp1

# time[istart: ifinal]
# orig_simulation_f[istart]

with PdfPages(outputfile) as pdf:
    for i in np.arange(istart, ifinal):  # np.arange(istart, istart + 5):  #
        # i = istart
        
        ################################ extract vorticity, wind
        
        rvor = rvor_100m.relative_vorticity[i, 80:920, 80:920].values * 10**4
        wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
        wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
        
        ######## theta
        orig_simulation = xr.open_dataset(orig_simulation_f[i])
        pres = orig_simulation.PS[0, 80:920, 80:920].values
        tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
        theta = tem2m * (p0sl/pres)**(r/cp)
        
        ################################ plot rvor
        fig, ax = framework_plot1(
            "1km_lb",
            plot_vorticity=True,
            xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
            vorticity_elements={
                'rvor': rvor,
                'lon': lon,
                'lat': lat,
                'vorlevel': np.arange(-12, 12.1, 0.1),
                'ticks': np.arange(-12, 12.1, 3),
                'time_point': time[i],
                'time_location': [-23, 34], },
            dpi=300
        )
        
        ################################ extract vortices information
        
        if decade_plot:
            indices = i
        else:
            indices = i - istart
        
        is_vortex = experiment.vortex_info.cols.is_vortex[indices]
        vortex_indices = \
            np.ma.array(experiment.vortex_info.cols.vortex_indices[indices])
        vortex_indices[is_vortex == 0] = np.ma.masked
        vortex_count = experiment.vortex_info.cols.vortex_count[indices]
        for j in range(vortex_count):
            # j = 0
            ################################ extract individual vortex info
            #### vortex points
            vortex_points = np.where(vortex_indices == j)
            vortex_points_center = (
                vortex_points[0].mean(), vortex_points[1].mean())
            from_maderira = poly_path.contains_point(vortex_points_center)
            if from_maderira:
                ################################ vortex detailed info
                vortex_de_r = [row[:]
                              for row in experiment.vortex_de_info.where(
                                  '(time_index == ' + str(i) +
                                  ') & (vortex_index == ' + str(j) + ')')][0]
                vortex_de_cols = experiment.vortex_de_info.colnames
                center_lat = vortex_de_r[vortex_de_cols.index('center_lat')]
                center_lon = vortex_de_r[vortex_de_cols.index('center_lon')]
                vortex_radius = vortex_de_r[vortex_de_cols.index('radius')]
                
                ################################ plot vortex mean winds
                vortex_wind_u = vortex_de_r[vortex_de_cols.index(
                    'mean_wind_u')]
                vortex_wind_v = vortex_de_r[vortex_de_cols.index(
                    'mean_wind_v')]
                ax.quiver(center_lon, center_lat, vortex_wind_u, vortex_wind_v,
                          rasterized=True)
                
                ################################ plot a nominal circle
                vortex_circle = plt.Circle(
                    (center_lon, center_lat), vortex_radius/1.1*0.01,
                    edgecolor='lime', facecolor = 'None', lw = 0.3, zorder = 2)
                ax.add_artist(vortex_circle)
                
                ################################ plot text
                ax.text(
                    center_lon, center_lat,
                    str(j) + ':' +
                    str(int(len(vortex_points[0]) * 1.2)) + ':' +
                    str(int(vortex_de_r[vortex_de_cols.index('angle')])) + ':' +
                    str(round(vortex_de_r[vortex_de_cols.index(
                        'distance2radius')], 1)),
                    color='m', size=6, fontweight='normal')
            # else:
            #     is_vortex[vortex_points] = 0
        
        ################################ plot identified vortices
        ax.contour(lon, lat, is_vortex,
                   colors='lime', levels=np.array([-0.5, 0.5]),
                   linewidths=0.3, linestyles='solid'
                   )
        
        ################################ plot local theta positive anomaly
        ax.contourf(
            lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]), zorder = 3)
        orig_simulation = xr.open_dataset(orig_simulation_f[i])
        pres = orig_simulation.PS[0, 80:920, 80:920].values
        tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
        theta = tem2m * (p0sl/pres)**(r/cp)
        find_plot_maxmin_points(
            lon, lat, data=theta, ax=ax, extrema='max', nsize=50,
            pointcolor='b', pointsize=1)
        pdf.savefig(fig)
        plt.close('all')
        print(str(i) + '/' + str(len(time)))


identified_rvor.close()


# endregion
# =============================================================================


# =============================================================================
# region check results across different simulation

inputfile0 = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09.h5"
inputfile1 = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5"
inputfile2 = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20.h5"
inputfile3 = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100214_20_stricter_dir.h5"

identified_rvor0 = tb.open_file(inputfile0, mode="r")
identified_rvor1 = tb.open_file(inputfile1, mode="r")
identified_rvor2 = tb.open_file(inputfile2, mode="r")
identified_rvor3 = tb.open_file(inputfile3, mode="r")

identified_rvor0.root.exp1.vortex_de_info.nrows
identified_rvor1.root.exp1.vortex_de_info.nrows
identified_rvor2.root.exp1.vortex_de_info.nrows
identified_rvor3.root.exp1.vortex_de_info.nrows

# endregion
# =============================================================================


# =============================================================================
# region check fpr
from DEoAI_analysis.module.vortex_namelist import correctly_reidentified

inputfile = "scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5"
identified_rvor = tb.open_file(inputfile, mode="r")

identified_count = 0
for i in correctly_reidentified.keys():
    identified_count += len(correctly_reidentified[i])

1 - identified_count/identified_rvor.root.exp1.vortex_de_info.nrows

# endregion
# =============================================================================


