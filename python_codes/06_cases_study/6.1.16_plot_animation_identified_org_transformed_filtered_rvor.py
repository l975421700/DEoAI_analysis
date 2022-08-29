

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
# region load common data

################################ load data
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201008.nc'
)
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values

################################ vortex identification
######## model topograph
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values

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
              years[iyear] + '*[0-9].nc')
))

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

######## mask topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-05 12:00 UTC

################################ original rvor and identification

timepoint = 73 + 68 - 33
rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
################################ import 3d constant file and original simulation
i = np.where(
    wind_100m.time == np.datetime64('2010-08-05T12:00:00.000000000'))[0][0]

wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values

######## theta
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

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(
    lon, lat, model_topo_mask,
    colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.01 identified vortices on original rvor_20100805_12.png')


# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-05 11:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-05T11:00:00.000000000'))[0][0]
timepoint = 73 + 68 - 34

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values

######## theta
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


fig, ax = framework_plot1("1km_lb", figsize=np.array([8.8, 9.3]) / 2.54,)

rvor_level = np.arange(-6, 6.01, 0.1)
rvor_ticks = np.arange(-6, 6.1, 2)
rvor_top = cm.get_cmap('Blues_r', int(np.floor(len(rvor_level) / 2)))
rvor_bottom = cm.get_cmap('Reds', int(np.floor(len(rvor_level) / 2)))
rvor_colors = np.vstack(
    (rvor_top(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2)))),
     [1, 1, 1, 1],
     rvor_bottom(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2))))))
rvor_cmp = ListedColormap(rvor_colors, name='RedsBlues_r')

plt_rvor = ax.pcolormesh(
    lon, lat, rvor, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),
    zorder=-2,)

cbar = fig.colorbar(
    plt_rvor, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=rvor_ticks, extend='both',
    anchor=(0.5, 1.5), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    '100-meter relative vorticity [$10^{-4}\;s^{-1}$]' + '\nat' + \
    ' ' + str(time[timepoint])[0:10] + ' ' + \
    str(time[timepoint])[11:13] + ':00 UTC', fontsize=10)

ax.contour(
    lon, lat, is_vortex1, colors='lime', levels=np.array([-0.5, 0.5]),
    linewidths=0.5, linestyles='solid',)
ax.contourf(
    lon, lat, model_topo_mask,
    colors='white', levels=np.array([0.5, 1.5]))

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.02 identified vortices on transformed rvor_20100805_11.png', dpi=600)

'''
'''

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-06 21:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification

(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_org,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.0.0 original rvor identification 2010080621.png')


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rec_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
    )
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.0.1 wavelet transformed rvor identification 2010080621.png')


################################ median filtered rvor

fil_rvor = median_filter(rvor, 3, )

(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.0.2 median filtered rvor identification 2010080621.png')


################################ median filtered rvor twice

fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )

(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor2,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil2,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.0.3 twice median filtered rvor identification 2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-04 08:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-04T08:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-04T08:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification

(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_org,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.1.0 original rvor identification 2010080408.png')


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rec_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.1.1 wavelet transformed rvor identification 2010080408.png')


################################ median filtered rvor

fil_rvor = median_filter(rvor, 3, )

(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.1.2 median filtered rvor identification 2010080408.png')


################################ median filtered rvor twice

fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )

(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor2,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil2,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.1.3 twice median filtered rvor identification 2010080408.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-07 21:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-07T21:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-07T21:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification

(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_org,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.2.0 original rvor identification 2010080721.png')


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rec_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.2.1 wavelet transformed rvor identification 2010080721.png')


################################ median filtered rvor

fil_rvor = median_filter(rvor, 3, )

(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.2.2 median filtered rvor identification 2010080721.png')


################################ median filtered rvor twice

fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )

(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor2,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil2,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.2.3 twice median filtered rvor identification 2010080721.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-08 05:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-08T05:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-08T05:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification

(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_org,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.3.0 original rvor identification 2010080805.png')


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rec_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.3.1 wavelet transformed rvor identification 2010080805.png')


################################ median filtered rvor

fil_rvor = median_filter(rvor, 3, )

(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.3.2 median filtered rvor identification 2010080805.png')


################################ median filtered rvor twice

fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )

(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor2,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil2,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.3.3 twice median filtered rvor identification 2010080805.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-08 12:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-08T12:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-08T12:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification

(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_org,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.4.0 original rvor identification 2010080812.png')


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rec_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.4.1 wavelet transformed rvor identification 2010080812.png')


################################ median filtered rvor

fil_rvor = median_filter(rvor, 3, )

(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.4.2 median filtered rvor identification 2010080812.png')


################################ median filtered rvor twice

fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )

(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor2,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil2,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.4.3 twice median filtered rvor identification 2010080812.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-04 04:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-04T04:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-04T04:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification

(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_org,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.5.0 original rvor identification 2010080404.png')


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rec_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.5.1 wavelet transformed rvor identification 2010080404.png')


################################ median filtered rvor

fil_rvor = median_filter(rvor, 3, )

(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.5.2 median filtered rvor identification 2010080404.png')


################################ median filtered rvor twice

fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )

(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': fil_rvor2,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_fil2,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.5.3 twice median filtered rvor identification 2010080404.png')

# endregion
# =============================================================================


# =============================================================================
# region 4 in 1 plot 2010-08-06 21:00

iy = np.where(
    wind_100m.time == np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
im = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    im, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[iy, 80:920, 80:920].values
wind_v = wind_100m.v_earth[iy, 80:920, 80:920].values

#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[iy])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification
(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

################################ median filtered rvor
fil_rvor = median_filter(rvor, 3, )
(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

################################ median filtered rvor twice
fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )
(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

nrow = 2
ncol = 2
fig = plt.figure(figsize=np.array([8.8*ncol, 7.5*nrow+1.8]) / 2.54, dpi = 300)
gs = fig.add_gridspec(nrow, ncol, wspace=0.15, hspace=0.15)
axs = gs.subplots(subplot_kw={'projection': transform})

for i in range(nrow):
    for j in range(ncol):
        
        axs[i, j].set_extent(extent1km_lb, crs=transform)
        axs[i, j].set_xticks(ticklabel1km_lb[0])
        axs[i, j].set_xticklabels(ticklabel1km_lb[1])
        axs[i, j].set_yticks(ticklabel1km_lb[2])
        axs[i, j].set_yticklabels(ticklabel1km_lb[3])
        axs[i, j].add_feature(coastline, lw=0.25)
        axs[i, j].add_feature(borders, lw=0.25)
        plt_gridline = axs[i, j].gridlines(
            crs=transform, linewidth=0.25,
            color='gray', alpha=0.5, linestyle='--')
        plt_gridline.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        plt_gridline.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        
        scale_bar(
            axs[i, j], bars=2, length=200, location=(0.04, 0.03),
            barheight=20, linewidth=0.15, col='black', middle_label=False)
        
        axs[i, j].text(
            -23, 34, str(wind_100m.time[iy].values)[0:10] + ' ' + \
            str(wind_100m.time[iy].values)[11:13] + ':00 UTC')
        
        if((i==0) & (j==0)):
            vorticity = rvor
            contour = is_vortex_org
            label = '(a)'
        elif((i==0) & (j==1)):
            vorticity = rec_rvor
            contour = is_vortex_trs
            label = '(b)'
        elif((i==1) & (j==0)):
            vorticity = fil_rvor
            contour = is_vortex_fil
            label = '(c)'
        elif((i==1) & (j==1)):
            vorticity = fil_rvor2
            contour = is_vortex_fil2
            label = '(d)'
        
        plt_rvor = axs[i, j].pcolormesh(
            lon, lat, vorticity, cmap=rvor_cmp, transform=transform,
            norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),)
        axs[i, j].contour(
            lon, lat, contour, colors='lime', levels=np.array([-0.5, 0.5]),
            linewidths=0.5, linestyles='solid')
        axs[i, j].text(-12.5, 34, label)

cbar = fig.colorbar(
    plt_rvor, ax = axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.5, aspect=25, anchor = (0.5, -0.15),
    ticks=rvor_ticks, extend='both')
cbar.ax.set_xlabel(
    "Identified original (a), wavelet transformed (b), median filtered (c) and twice median filtered (d) \n 100-meter relative vorticity [$10^{-4}\;s^{-1}$]")
# fig.tight_layout()
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.99)

fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.6.0 vortex identification with different preprocessing 2010080621.png')

# endregion
# =============================================================================


# =============================================================================
# region 4 in 1 animation 2010-08 03-09

iy_start = np.where(wind_100m.time == np.datetime64('2010-08-03T20'))[0][0]
iy_final = np.where(wind_100m.time == np.datetime64('2010-08-09T07'))[0][0]
im_start = np.where(
    rvorticity_1km_1h_100m.time == np.datetime64('2010-08-03T20'))[0][0]
im_final = np.where(
    rvorticity_1km_1h_100m.time == np.datetime64('2010-08-09T07'))[0][0]
outputfile = 'figures/06_case_study/06_05_org_transformed_filtered/6_5.7.0 vortex identification with different preprocessing_20100803_09.mp4'


nrow = 2
ncol = 2

fig = plt.figure(figsize=np.array([8.8*ncol, 7.5*nrow+1.8]) / 2.54, dpi=300)
gs = fig.add_gridspec(nrow, ncol, wspace=0.15, hspace=0.15)
axs = gs.subplots(subplot_kw={'projection': transform})
ims = []

for k in range(iy_final - iy_start + 1):  # range(2): #
    # k=0
    iy = iy_start + k
    im = im_start + k
    plt_rvor = [0, 0, 0, 0]
    plt_text = []
    plt_contour = []
    
    ################################ load data
    rvor = rvorticity_1km_1h_100m.relative_vorticity[
        im, 80:920, 80:920].values * 10**4
    wind_u = wind_100m.u_earth[iy, 80:920, 80:920].values
    wind_v = wind_100m.v_earth[iy, 80:920, 80:920].values
    #### theta
    orig_simulation = xr.open_dataset(orig_simulation_f[iy])
    pres = orig_simulation.PS[0, 80:920, 80:920].values
    tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
    theta = tem2m * (p0sl/pres)**(r/cp)
    
    ################################ vortex identification
    #### original rvor identification
    (vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
     theta_anomalies_org
     ) = vortex_identification1(
        rvor, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
    
    #### wavelet transformed rvor
    coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
    n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
    (vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
     theta_anomalies_trs
     ) = vortex_identification1(
        rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
    
    #### median filtered rvor
    fil_rvor = median_filter(rvor, 3, )
    (vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
     theta_anomalies_fil
     ) = vortex_identification1(
        fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
     
    #### median filtered rvor twice
    fil_rvor = median_filter(rvor, 3, )
    fil_rvor2 = median_filter(fil_rvor, 3, )
    (vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
     theta_anomalies_fil2
     ) = vortex_identification1(
        fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
    
    
    for i in range(nrow):
        for j in range(ncol):
            if((i == 0) & (j == 0)):
                vorticity = rvor
                contour = is_vortex_org
            elif((i == 0) & (j == 1)):
                vorticity = rec_rvor
                contour = is_vortex_trs
            elif((i == 1) & (j == 0)):
                vorticity = fil_rvor
                contour = is_vortex_fil
            elif((i == 1) & (j == 1)):
                vorticity = fil_rvor2
                contour = is_vortex_fil2
            
            plt_rvor[i*2 + j] = axs[i, j].pcolormesh(
                lon, lat, vorticity, cmap=rvor_cmp, transform=transform,
                norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),)
            
            contours = axs[i, j].contour(
                lon, lat, contour, colors='lime', levels=np.array([-0.5, 0.5]),
                linewidths=0.5, linestyles='solid')
            contours.__class__ = mpl.contour.QuadContourSet
            plt_contour += contours.collections
            
            timestamp = axs[i, j].text(
                -23, 34, str(wind_100m.time[iy].values)[0:10] + ' ' +
                str(wind_100m.time[iy].values)[11:13] + ':00 UTC')
            plt_text.append(timestamp)
    
    ims.append(plt_rvor + plt_contour + plt_text)
    print(str(k) + '/' + str(iy_final - iy_start))

for i in range(nrow):
    for j in range(ncol):
        axs[i, j].set_extent(extent1km_lb, crs=transform)
        axs[i, j].set_xticks(ticklabel1km_lb[0])
        axs[i, j].set_xticklabels(ticklabel1km_lb[1])
        axs[i, j].set_yticks(ticklabel1km_lb[2])
        axs[i, j].set_yticklabels(ticklabel1km_lb[3])
        axs[i, j].add_feature(coastline, lw=0.25)
        axs[i, j].add_feature(borders, lw=0.25)
        plt_gridline = axs[i, j].gridlines(
            crs=transform, linewidth=0.25,
            color='gray', alpha=0.5, linestyle='--')
        plt_gridline.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        plt_gridline.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        scale_bar(
            axs[i, j], bars=2, length=200, location=(0.04, 0.03),
            barheight=20, linewidth=0.15, col='black', middle_label=False)
        
        if((i==0) & (j==0)):
            label = '(a)'
        elif((i==0) & (j==1)):
            label = '(b)'
        elif((i==1) & (j==0)):
            label = '(c)'
        elif((i==1) & (j==1)):
            label = '(d)'
        axs[i, j].text(-12.5, 34, label)

cbar = fig.colorbar(
    plt_rvor[0], ax = axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.5, aspect=25, anchor = (0.5, -0.15),
    ticks=rvor_ticks, extend='both')
cbar.ax.set_xlabel(
    "Identified original (a), wavelet transformed (b), median filtered (c) and twice median filtered (d) \n 100-meter relative vorticity [$10^{-4}\;s^{-1}$]")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


# endregion
# =============================================================================


# =============================================================================
# region 4 in 1 animation 2010-02 14-22

rvorticity_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201002.nc'
)

iy_start = np.where(wind_100m.time == np.datetime64('2010-02-14T00'))[0][0]
iy_final = np.where(wind_100m.time == np.datetime64('2010-02-22T00'))[0][0]
im_start = np.where(
    rvorticity_100m.time == np.datetime64('2010-02-14T00'))[0][0]
im_final = np.where(
    rvorticity_100m.time == np.datetime64('2010-02-22T00'))[0][0]
outputfile = 'figures/06_case_study/06_05_org_transformed_filtered/6_5.7.1 vortex identification with different preprocessing_20100214_22.mp4'


nrow = 2
ncol = 2

fig = plt.figure(figsize=np.array([8.8*ncol, 7.5*nrow+1.8]) / 2.54, dpi=300)
gs = fig.add_gridspec(nrow, ncol, wspace=0.15, hspace=0.15)
axs = gs.subplots(subplot_kw={'projection': transform})
ims = []

for k in range(iy_final - iy_start + 1):  # range(2):  #
    # k=0
    iy = iy_start + k
    im = im_start + k
    plt_rvor = [0, 0, 0, 0]
    plt_text = []
    plt_contour = []
    
    ################################ load data
    rvor = rvorticity_100m.relative_vorticity[
        im, 80:920, 80:920].values * 10**4
    wind_u = wind_100m.u_earth[iy, 80:920, 80:920].values
    wind_v = wind_100m.v_earth[iy, 80:920, 80:920].values
    #### theta
    orig_simulation = xr.open_dataset(orig_simulation_f[iy])
    pres = orig_simulation.PS[0, 80:920, 80:920].values
    tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
    theta = tem2m * (p0sl/pres)**(r/cp)
    
    ################################ vortex identification
    #### original rvor identification
    (vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
     theta_anomalies_org
     ) = vortex_identification1(
        rvor, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
    
    #### wavelet transformed rvor
    coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
    n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
    (vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
     theta_anomalies_trs
     ) = vortex_identification1(
        rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
    
    #### median filtered rvor
    fil_rvor = median_filter(rvor, 3, )
    (vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
     theta_anomalies_fil
     ) = vortex_identification1(
        fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
     
    #### median filtered rvor twice
    fil_rvor = median_filter(rvor, 3, )
    fil_rvor2 = median_filter(fil_rvor, 3, )
    (vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
     theta_anomalies_fil2
     ) = vortex_identification1(
        fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
        center_madeira, poly_path, madeira_mask,
        min_rvor, min_max_rvor, min_size, min_size_theta,
        min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
        max_distance2radius,
        original_rvorticity=rvor, reject_info=False,
        grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
    )
    
    
    for i in range(nrow):
        for j in range(ncol):
            if((i == 0) & (j == 0)):
                vorticity = rvor
                contour = is_vortex_org
            elif((i == 0) & (j == 1)):
                vorticity = rec_rvor
                contour = is_vortex_trs
            elif((i == 1) & (j == 0)):
                vorticity = fil_rvor
                contour = is_vortex_fil
            elif((i == 1) & (j == 1)):
                vorticity = fil_rvor2
                contour = is_vortex_fil2
            
            plt_rvor[i*2 + j] = axs[i, j].pcolormesh(
                lon, lat, vorticity, cmap=rvor_cmp, transform=transform,
                norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),)
            
            contours = axs[i, j].contour(
                lon, lat, contour, colors='lime', levels=np.array([-0.5, 0.5]),
                linewidths=0.5, linestyles='solid')
            contours.__class__ = mpl.contour.QuadContourSet
            plt_contour += contours.collections
            
            timestamp = axs[i, j].text(
                -23, 34, str(wind_100m.time[iy].values)[0:10] + ' ' +
                str(wind_100m.time[iy].values)[11:13] + ':00 UTC')
            plt_text.append(timestamp)
    
    ims.append(plt_rvor + plt_contour + plt_text)
    print(str(k) + '/' + str(iy_final - iy_start))

for i in range(nrow):
    for j in range(ncol):
        axs[i, j].set_extent(extent1km_lb, crs=transform)
        axs[i, j].set_xticks(ticklabel1km_lb[0])
        axs[i, j].set_xticklabels(ticklabel1km_lb[1])
        axs[i, j].set_yticks(ticklabel1km_lb[2])
        axs[i, j].set_yticklabels(ticklabel1km_lb[3])
        axs[i, j].add_feature(coastline, lw=0.25)
        axs[i, j].add_feature(borders, lw=0.25)
        plt_gridline = axs[i, j].gridlines(
            crs=transform, linewidth=0.25,
            color='gray', alpha=0.5, linestyle='--')
        plt_gridline.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        plt_gridline.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        scale_bar(
            axs[i, j], bars=2, length=200, location=(0.04, 0.03),
            barheight=20, linewidth=0.15, col='black', middle_label=False)
        
        if((i==0) & (j==0)):
            label = '(a)'
        elif((i==0) & (j==1)):
            label = '(b)'
        elif((i==1) & (j==0)):
            label = '(c)'
        elif((i==1) & (j==1)):
            label = '(d)'
        axs[i, j].text(-12.5, 34, label)

cbar = fig.colorbar(
    plt_rvor[0], ax = axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.5, aspect=25, anchor = (0.5, -0.15),
    ticks=rvor_ticks, extend='both')
cbar.ax.set_xlabel(
    "Identified original (a), wavelet transformed (b), median filtered (c) and twice median filtered (d) \n 100-meter relative vorticity [$10^{-4}\;s^{-1}$]")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# =============================================================================


# =============================================================================
# region 4 in 1 plot 2010-02-17 20 UTC

rvorticity_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201002.nc')

iy = np.where(wind_100m.time == np.datetime64('2010-02-17T20'))[0][0]
im = np.where(rvorticity_100m.time == np.datetime64('2010-02-17T20'))[0][0]

rvor = rvorticity_100m.relative_vorticity[
    im, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[iy, 80:920, 80:920].values
wind_v = wind_100m.v_earth[iy, 80:920, 80:920].values

#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[iy])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ original rvor identification
(vortices_org, is_vortex_org, vortices_count_org, vortex_indices_org,
 theta_anomalies_org
 ) = vortex_identification1(
    rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

################################ median filtered rvor
fil_rvor = median_filter(rvor, 3, )
(vortices_fil, is_vortex_fil, vortices_count_fil, vortex_indices_fil,
 theta_anomalies_fil
 ) = vortex_identification1(
    fil_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

################################ median filtered rvor twice
fil_rvor = median_filter(rvor, 3, )
fil_rvor2 = median_filter(fil_rvor, 3, )
(vortices_fil2, is_vortex_fil2, vortices_count_fil2, vortex_indices_fil2,
 theta_anomalies_fil2
 ) = vortex_identification1(
    fil_rvor2, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,)

nrow = 2
ncol = 2
fig = plt.figure(figsize=np.array([8.8*ncol, 7.5*nrow+1.8]) / 2.54, dpi = 300)
gs = fig.add_gridspec(nrow, ncol, wspace=0.15, hspace=0.15)
axs = gs.subplots(subplot_kw={'projection': transform})

for i in range(nrow):
    for j in range(ncol):
        
        axs[i, j].set_extent(extent1km_lb, crs=transform)
        axs[i, j].set_xticks(ticklabel1km_lb[0])
        axs[i, j].set_xticklabels(ticklabel1km_lb[1])
        axs[i, j].set_yticks(ticklabel1km_lb[2])
        axs[i, j].set_yticklabels(ticklabel1km_lb[3])
        axs[i, j].add_feature(coastline, lw=0.25)
        axs[i, j].add_feature(borders, lw=0.25)
        plt_gridline = axs[i, j].gridlines(
            crs=transform, linewidth=0.25,
            color='gray', alpha=0.5, linestyle='--')
        plt_gridline.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
        plt_gridline.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
        
        scale_bar(
            axs[i, j], bars=2, length=200, location=(0.02, 0.015),
            barheight=20, linewidth=0.15, col='black', middle_label=False)
        
        axs[i, j].text(
            -23, 34, str(wind_100m.time[iy].values)[0:10] + ' ' + \
            str(wind_100m.time[iy].values)[11:13] + ':00 UTC')
        
        if((i==0) & (j==0)):
            vorticity = rvor
            contour = is_vortex_org
            label = '(a)'
        elif((i==0) & (j==1)):
            vorticity = rec_rvor
            contour = is_vortex_trs
            label = '(b)'
        elif((i==1) & (j==0)):
            vorticity = fil_rvor
            contour = is_vortex_fil
            label = '(c)'
        elif((i==1) & (j==1)):
            vorticity = fil_rvor2
            contour = is_vortex_fil2
            label = '(d)'
        
        plt_rvor = axs[i, j].pcolormesh(
            lon, lat, vorticity, cmap=rvor_cmp, transform=transform,
            norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),)
        axs[i, j].contour(
            lon, lat, contour, colors='lime', levels=np.array([-0.5, 0.5]),
            linewidths=0.5, linestyles='solid')
        axs[i, j].text(-12.5, 34, label)
        axs[i, j].contourf(lon, lat, model_topo_mask,
                    colors='white', levels=np.array([0.5, 1.5]))

cbar = fig.colorbar(
    plt_rvor, ax = axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.5, aspect=25, anchor = (0.5, -0.15),
    ticks=rvor_ticks, extend='both')
cbar.ax.set_xlabel(
    "Identified (a) original, (b) wavelet transformed, (c) median filtered and (d) twice median filtered \n 100-meter relative vorticity [$10^{-4}\;s^{-1}$]")
# fig.tight_layout()
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.99)

fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.6.1 vortex identification with different preprocessing 2010021720.png')

# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-14 12:00 UTC

dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

i = np.where(
    wind_100m.time == np.datetime64('2010-08-14T12:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-14T12:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)


################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

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
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid')
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))

fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.8.0 wavelet transformed rvor identification 2010081412.png')


# endregion
# =============================================================================


# =============================================================================
# region plot vortices for 2010-08-06 21:00 UTC

i = np.where(
    wind_100m.time == np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
timepoint = np.where(
    rvorticity_1km_1h_100m.time ==
    np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]

rvor = rvorticity_1km_1h_100m.relative_vorticity[
    timepoint, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
#### theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

################################ wavelet transformed rvor
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)

(vortices_trs, is_vortex_trs, vortices_count_trs, vortex_indices_trs,
 theta_anomalies_trs
 ) = vortex_identification1(
    rec_rvor, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity=rvor, reject_info=False,
    grid_size=1.2, median_filter_size=3, maximum_filter_size=50,
)

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': time[timepoint], 'time_location': [-23, 34], },
)
ax.contour(lon, lat, is_vortex_trs,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.5, linestyles='solid'
           )
ax.contourf(lon, lat, model_topo_mask,
            colors='white', levels=np.array([0.5, 1.5]))
fig.savefig(
    'figures/06_case_study/06_05_org_transformed_filtered/6_5.8.1 wavelet transformed rvor identification 2010080621.png')


# endregion
# =============================================================================


