

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

# region create ellipse mask

nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')

lon = nc3d_lb_c.lon.values
lat = nc3d_lb_c.lat.values

from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1


mask_e1 = ellipse(
    center_madeira[0],
    center_madeira[1],
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )
mask_e2 = ellipse(
    center_madeira[0] + radius_madeira[0] * 3 * np.cos(
        np.deg2rad(angle_deg_madeira)),
    center_madeira[1] + radius_madeira[0] * 3 * np.sin(
        np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )
mask_e3 = ellipse(
    center_madeira[0] + radius_madeira[1] * 5 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    center_madeira[1] - radius_madeira[1] * 5 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )


# endregion
# =============================================================================


# =============================================================================
# region import data


################################ inversion base height
inversion_height_20100801_09 = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm_post_processed/inversion_height_20100801_09.nc'
    )
inversion_height = inversion_height_20100801_09.inversion_height[
    68:200, :, :].values
inversion_height_e2 = np.ones((inversion_height.shape[0], np.sum(mask_e2)))
for i in np.arange(0, inversion_height.shape[0]):
    inversion_height_e2[i, :] = inversion_height[i, mask_e2].flatten()


################################ 100m vorticity
iyear = 4
# years[iyear]
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values
lon = rvor_100m.lon[80:920, 80:920].values
lat = rvor_100m.lat[80:920, 80:920].values


################################ wind velocity in e3
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
ifinal = np.where(time == np.datetime64('2010-08-09T08:00:00.000000000'))[0][0]
u_earth = wind_100m.u_earth[istart:ifinal, 80:920, 80:920].values
v_earth = wind_100m.v_earth[istart:ifinal, 80:920, 80:920].values
u_earth_e3 = np.ones((u_earth.shape[0], np.sum(mask_e3)))
v_earth_e3 = np.ones((u_earth.shape[0], np.sum(mask_e3)))
for i in np.arange(0, u_earth.shape[0]):
    u_earth_e3[i, :] = u_earth[i, mask_e3].flatten()
    v_earth_e3[i, :] = v_earth[i, mask_e3].flatten()


################################ topography
target_grid = xe.util.grid_2d(
    lon0_b=lon.min(), lon1_b=lon.max(), d_lon=0.01,
    lat0_b=lat.min(), lat1_b=lat.max(), d_lat=0.01)
nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
regridder_c = xe.Regridder(
    nc3d_lb_c, target_grid, 'bilinear', reuse_weights=True)
hsurf = regridder_c(nc3d_lb_c.HSURF.squeeze())

# regrid
ds = xr.merge([hsurf], compat='override')
dset_cross = ds.metpy.parse_cf()
dset_cross['y'] = dset_cross['lat'].values[:, 0]
dset_cross['x'] = dset_cross['lon'].values[0, :]

# cross section
from geopy import distance
# B1
startpoint_c = [
    center_madeira[1] + 1.2 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
    center_madeira[0] - 1.2 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
    ]
# B2
endpoint_c = [
    center_madeira[1] - 2 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
    center_madeira[0] + 2 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira + 0)),
    ]
cross_section_distance1 = distance.distance(startpoint_c, endpoint_c).km
cross = cross_section(
    dset_cross,
    startpoint_c,
    endpoint_c,
    steps=int(cross_section_distance1/1.1)+1,
).set_coords(('y', 'x'))
# sum(cross['HSURF'].values > 800)


################################ tracked vortices
tracked_rvor = tb.open_file(
    'scratch/rvorticity/rvor_track/identified_transformed_rvor_20100803_09_track_4p_4n.h5',
    mode="r")
group_name = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']
# endregion
# =============================================================================


dist12 = np.zeros((6))
time12 = np.zeros((6))
hinv = np.zeros((6))
mean_wind = np.zeros((6))
crosswind_d = np.zeros((6))
wind_e3 = np.zeros((6))
st = np.zeros((6))


# =============================================================================
# region calculate Strauhal number between P2 and P1
i = 0
former = 'P1'
later = 'P2'

iformer = np.where(time == np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
ilater = np.where(time == np.datetime64('2010-08-07T03:00:00.000000000'))[0][0]
# time[iformer:(ilater + 1)]

################################ distance
former_lat = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lat[ilater - iformer]
former_lon = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lon[ilater - iformer]
later_lat = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lat[0]
later_lon = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lon[0]
dist12[i] = distance.distance(
    [later_lat, later_lon], [former_lat, former_lon]).m

################################ wind velocity
mean_wind_u = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_u[:(ilater - iformer + 1)]
mean_wind_v = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_v[:(ilater - iformer + 1)]
mean_wind[i] = (np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)).mean()

################################ shedding period
time12[i] = dist12[i] / mean_wind[i]  # in hours
# 6.080 h

################################ inversion base height and crosswind diameter
hinv[i] = inversion_height_e2[(iformer - istart):(ilater - istart + 1), ].mean()
crosswind_d[i] = sum(cross['HSURF'].values > hinv[i]) * 1100 # in [m]
# 35200 m
# tracked_rvor.close()

################################ undisturbed upstream wind velocity
wind_e3[i] = np.sqrt(
    u_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2 +
    v_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2
)

################################ Strouhal number
st[i] = crosswind_d[i] / (time12[i] * wind_e3[i])

'''
tracked_rvor.root[
    former].extracted_vortex_info.cols.time_index[:(ilater - iformer + 1)]
'''
# endregion
# =============================================================================


# =============================================================================
# region calculate Strauhal number between P3 and P2
i = 1
former = 'P2'
later = 'P3'

iformer = np.where(time == np.datetime64(
    '2010-08-07T03:00:00.000000000'))[0][0]
ilater = np.where(time == np.datetime64('2010-08-07T09:00:00.000000000'))[0][0]
# time[iformer:(ilater + 1)]

################################ distance
former_lat = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lat[ilater - iformer]
former_lon = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lon[ilater - iformer]
later_lat = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lat[0]
later_lon = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lon[0]
dist12[i] = distance.distance(
    [later_lat, later_lon], [former_lat, former_lon]).m

################################ wind velocity
mean_wind_u = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_u[:(ilater - iformer + 1)]
mean_wind_v = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_v[:(ilater - iformer + 1)]
mean_wind[i] = (np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)).mean()

################################ shedding period
time12[i] = dist12[i] / mean_wind[i]  # in hours

################################ inversion base height and crosswind diameter
hinv[i] = inversion_height_e2[(iformer - istart)
                               :(ilater - istart + 1), ].mean()
crosswind_d[i] = sum(cross['HSURF'].values > hinv[i]) * 1100  # in [m]
# 35200 m
# tracked_rvor.close()

################################ undisturbed upstream wind velocity
wind_e3[i] = np.sqrt(
    u_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2 +
    v_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2
)

################################ Strouhal number
st[i] = crosswind_d[i] / (time12[i] * wind_e3[i])

# endregion
# =============================================================================


# =============================================================================
# region calculate Strauhal number between P4 and P3
i = 2
former = 'P3'
later = 'P4'

iformer = np.where(time == np.datetime64(
    '2010-08-07T09:00:00.000000000'))[0][0]
ilater = np.where(time == np.datetime64('2010-08-07T16:00:00.000000000'))[0][0]
# time[iformer:(ilater + 1)]

################################ distance
former_lat = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lat[ilater - iformer]
former_lon = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lon[ilater - iformer]
later_lat = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lat[0]
later_lon = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lon[0]
dist12[i] = distance.distance(
    [later_lat, later_lon], [former_lat, former_lon]).m

################################ wind velocity
mean_wind_u = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_u[:(ilater - iformer + 1)]
mean_wind_v = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_v[:(ilater - iformer + 1)]
mean_wind[i] = (np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)).mean()

################################ shedding period
time12[i] = dist12[i] / mean_wind[i]  # in hours

################################ inversion base height and crosswind diameter
hinv[i] = inversion_height_e2[(iformer - istart)
                               :(ilater - istart + 1), ].mean()
crosswind_d[i] = sum(cross['HSURF'].values > hinv[i]) * 1100  # in [m]
# 35200 m
# tracked_rvor.close()

################################ undisturbed upstream wind velocity
wind_e3[i] = np.sqrt(
    u_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2 +
    v_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2
)

################################ Strouhal number
st[i] = crosswind_d[i] / (time12[i] * wind_e3[i])

# endregion
# =============================================================================


# =============================================================================
# region calculate Strauhal number between N2 and N1
i = 3
former = 'N1'
later = 'N2'

iformer = np.where(time == np.datetime64(
    '2010-08-07T00:00:00.000000000'))[0][0]
ilater = np.where(time == np.datetime64('2010-08-07T06:00:00.000000000'))[0][0]
# time[iformer:(ilater + 1)]

################################ distance
former_lat = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lat[ilater - iformer]
former_lon = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lon[ilater - iformer]
later_lat = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lat[0]
later_lon = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lon[0]
dist12[i] = distance.distance(
    [later_lat, later_lon], [former_lat, former_lon]).m

################################ wind velocity
mean_wind_u = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_u[:(ilater - iformer + 1)]
mean_wind_v = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_v[:(ilater - iformer + 1)]
mean_wind[i] = (np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)).mean()

################################ shedding period
time12[i] = dist12[i] / mean_wind[i]  # in hours

################################ inversion base height and crosswind diameter
hinv[i] = inversion_height_e2[(iformer - istart)
                               :(ilater - istart + 1), ].mean()
crosswind_d[i] = sum(cross['HSURF'].values > hinv[i]) * 1100  # in [m]
# 35200 m
# tracked_rvor.close()

################################ undisturbed upstream wind velocity
wind_e3[i] = np.sqrt(
    u_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2 +
    v_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2
)

################################ Strouhal number
st[i] = crosswind_d[i] / (time12[i] * wind_e3[i])

# endregion
# =============================================================================


# =============================================================================
# region calculate Strauhal number between N2 and N1
i = 4
former = 'N2'
later = 'N3'

iformer = np.where(time == np.datetime64(
    '2010-08-07T06:00:00.000000000'))[0][0]
ilater = np.where(time == np.datetime64('2010-08-07T12:00:00.000000000'))[0][0]
# time[iformer:(ilater + 1)]

################################ distance
former_lat = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lat[ilater - iformer]
former_lon = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lon[ilater - iformer]
later_lat = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lat[0]
later_lon = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lon[0]
dist12[i] = distance.distance(
    [later_lat, later_lon], [former_lat, former_lon]).m

################################ wind velocity
mean_wind_u = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_u[:(ilater - iformer + 1)]
mean_wind_v = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_v[:(ilater - iformer + 1)]
mean_wind[i] = (np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)).mean()

################################ shedding period
time12[i] = dist12[i] / mean_wind[i]  # in hours

################################ inversion base height and crosswind diameter
hinv[i] = inversion_height_e2[(iformer - istart)
                               :(ilater - istart + 1), ].mean()
crosswind_d[i] = sum(cross['HSURF'].values > hinv[i]) * 1100  # in [m]
# 35200 m
# tracked_rvor.close()

################################ undisturbed upstream wind velocity
wind_e3[i] = np.sqrt(
    u_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2 +
    v_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2
)

################################ Strouhal number
st[i] = crosswind_d[i] / (time12[i] * wind_e3[i])

# endregion
# =============================================================================


# =============================================================================
# region calculate Strauhal number between N2 and N1
i = 5
former = 'N3'
later = 'N4'

iformer = np.where(time == np.datetime64(
    '2010-08-07T12:00:00.000000000'))[0][0]
ilater = np.where(time == np.datetime64('2010-08-07T16:00:00.000000000'))[0][0]
# time[iformer:(ilater + 1)]

################################ distance
former_lat = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lat[ilater - iformer]
former_lon = tracked_rvor.root[
    former].extracted_vortex_info.cols.center_lon[ilater - iformer]
later_lat = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lat[0]
later_lon = tracked_rvor.root[
    later].extracted_vortex_info.cols.center_lon[0]
dist12[i] = distance.distance(
    [later_lat, later_lon], [former_lat, former_lon]).m

################################ wind velocity
mean_wind_u = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_u[:(ilater - iformer + 1)]
mean_wind_v = tracked_rvor.root[
    former].extracted_vortex_info.cols.mean_wind_v[:(ilater - iformer + 1)]
mean_wind[i] = (np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)).mean()

################################ shedding period
time12[i] = dist12[i] / mean_wind[i]  # in hours

################################ inversion base height and crosswind diameter
hinv[i] = inversion_height_e2[(iformer - istart)
                               :(ilater - istart + 1), ].mean()
crosswind_d[i] = sum(cross['HSURF'].values > hinv[i]) * 1100  # in [m]
# 35200 m
# tracked_rvor.close()

################################ undisturbed upstream wind velocity
wind_e3[i] = np.sqrt(
    u_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2 +
    v_earth_e3[(iformer - istart):(ilater - istart + 1), ].mean() ** 2
)

################################ Strouhal number
st[i] = crosswind_d[i] / (time12[i] * wind_e3[i])

# endregion
# =============================================================================


'''
In [322]: time12/3600
Out[322]: array([6.08019941, 6.23318265, 7.77958732, 6.57795965, 6.92325013,
       6.86827704])
In [323]: np.mean(time12/3600)
Out[323]: 6.743742701391351

In [324]: hinv
Out[324]: array([793.43979284, 833.59948209, 806.5521148 , 803.86275356,
       838.06646526, 792.08459215])
In [325]: mean_wind
Out[325]: array([7.633882  , 7.0164182 , 5.88520328, 6.82341177, 6.12525981,
       5.78057736])

crosswind_d = 35.2 km

In [327]: wind_e3
Out[327]: array([10.17417084,  9.80811469,  8.87945149, 10.03791239,  9.46754487,
        8.56947451])

In [328]: st
Out[328]: array([0.15806049, 0.15993546, 0.14154596, 0.14808312, 0.14917387,
       0.16612621])
In [331]: stats.describe(st)
Out[331]: DescribeResult(nobs=6, minmax=(0.1415459553621382, 0.16612621366698657), mean=0.15382085221172848, variance=8.239479230824071e-05, skewness=0.008607069400657722, kurtosis=-1.2712217436125062)

In [329]: np.mean(st)
Out[329]: 0.15382085221172848

In [330]: np.std(st)
Out[330]: 0.008286273403860987
'''
