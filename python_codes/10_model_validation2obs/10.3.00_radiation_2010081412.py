

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
# region load data

################################ import rvor
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201008.nc')
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values

################################ vortex identification
######## model topograph
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

######## parameter settings
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
reject_info = True

######## original simulation to calculate surface theta
iyear = 4
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')))

######## create a mask for Madeira
from matplotlib.path import Path
polygon=[
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
    ]
poly_path=Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)

######## import wind
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

# endregion
# =============================================================================


# =============================================================================
# region load time point data

timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-14T12:00:00.000000000'))[0][0]
i = np.where(
    wind_100m.time == np.datetime64('2010-08-14T12:00:00.000000000'))[0][0]

# time[timepoint + 68]
rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4

wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values

orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices1, is_vortex1, vortices_count1, vortex_indices1, theta_anomalies1) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)


# endregion
# =============================================================================


# =============================================================================
# region plot rvor

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },)
ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/10_validation2obs/10_02_2010081412/10_02.00 identified vortices on wavelet transformed rvor_20100814_12.png')

# endregion
# =============================================================================


# =============================================================================
# region plot theta

theta_min = 281 + 9
theta_mid = 293
theta_max = 305 - 9

theta_ticks = np.arange(theta_min, theta_max + 0.01, 1)
theta_level = np.arange(theta_min, theta_max + 0.01, 0.025)


fig, ax = framework_plot1(
    "1km_lb",
)

theta_time = ax.text(
    -23, 34, str(time[timepoint])[0:10] + ' ' + \
    str(time[timepoint])[11:13] + ':00 UTC')
plt_theta = ax.pcolormesh(
    lon, lat, theta, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_theta, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=theta_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Surface potential temperature [K]")

ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/10_validation2obs/10_02_2010081412/10_02.01 surface theta_20100814_12.png')

# endregion
# =============================================================================


